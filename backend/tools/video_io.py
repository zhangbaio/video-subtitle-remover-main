import queue
import subprocess
import threading
import time

import cv2
import numpy as np

from .ffmpeg_cli import FFmpegCLI
from .subprocess_utils import hidden_subprocess_kwargs


class FramePrefetcher:
    """
    后台线程预解码视频帧，使 I/O 与模型推理重叠。
    接口兼容 cv2.VideoCapture（read/release）。
    """

    def __init__(self, video_cap, buffer_size=10):
        self.cap = video_cap
        self._buffer = queue.Queue(maxsize=buffer_size)
        self._stopped = False
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        while not self._stopped:
            ret, frame = self.cap.read()
            self._buffer.put((ret, frame))
            if not ret:
                break

    def read(self):
        """读取下一帧，接口与 cv2.VideoCapture.read() 一致。"""
        return self._buffer.get()

    def get(self, propId):
        return self.cap.get(propId)

    def stop(self):
        """停止预读取，不释放底层 video_cap。"""
        self._stopped = True
        try:
            while not self._buffer.empty():
                self._buffer.get_nowait()
        except queue.Empty:
            pass
        self._thread.join(timeout=5)

    def release(self):
        self.stop()
        self.cap.release()


class FFmpegVideoWriter:
    """
    通过 FFmpeg 管道写入帧，使用 libx264 编码。
    接口兼容 cv2.VideoWriter（write/release）。
    """

    _QUEUE_POLL_SECONDS = 0.05
    _WRITE_TIMEOUT_SECONDS = 30.0
    _FINALIZE_TIMEOUT_SECONDS = 600.0
    _FORCE_STOP_TIMEOUT_SECONDS = 5.0
    _STOP = object()

    def __init__(self, output_path, fps, size, queue_size=16):
        w, h = size
        cmd = [
            FFmpegCLI.instance().ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
            '-c:v', 'libx264',
            '-pix_fmt', 'yuv420p',
            '-crf', '18',
            '-preset', 'fast',
            '-loglevel', 'error',
            output_path
        ]
        queue_size = int(queue_size)
        if queue_size < 1:
            raise ValueError('queue_size must be at least 1')

        self._released = False
        self._release_complete = False
        self._release_error = None
        self._output_path = output_path
        self._writer_error = None
        self._writer_error_event = threading.Event()
        # Serializes frame/sentinel insertion. This keeps a concurrent release
        # from placing the sentinel before a write that has already started.
        self._enqueue_lock = threading.Lock()
        self._release_lock = threading.Lock()
        self._frame_queue = queue.Queue(maxsize=queue_size)
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            **hidden_subprocess_kwargs(),
        )

        if self._process.stdin is None:
            try:
                self._process.terminate()
            except OSError:
                pass
            raise RuntimeError('FFmpeg video writer has no input pipe')

        self._writer_thread = threading.Thread(
            target=self._write_loop,
            name='ffmpeg-video-writer',
            daemon=True,
        )
        self._writer_thread.start()

    def _set_writer_error(self, error):
        if not self._writer_error_event.is_set():
            self._writer_error = error
            self._writer_error_event.set()

    def _raise_writer_error(self):
        if self._writer_error_event.is_set():
            raise self._writer_error

    def _write_loop(self):
        while True:
            item = self._frame_queue.get()
            try:
                if item is self._STOP:
                    return
                self._process.stdin.write(item)
            except BaseException as error:
                self._set_writer_error(error)
                return
            finally:
                self._frame_queue.task_done()

    @staticmethod
    def _remaining_seconds(deadline):
        return max(0.0, deadline - time.monotonic())

    def _put_until(self, item, deadline, timeout_error):
        while True:
            self._raise_writer_error()
            remaining = self._remaining_seconds(deadline)
            if remaining <= 0:
                raise timeout_error
            try:
                self._frame_queue.put(
                    item,
                    timeout=min(self._QUEUE_POLL_SECONDS, remaining),
                )
                return
            except queue.Full:
                continue

    def _force_stop_process(self):
        """Best-effort bounded shutdown used after a pipe timeout."""
        try:
            self._process.terminate()
        except (OSError, ValueError):
            pass
        try:
            self._process.wait(timeout=self._FORCE_STOP_TIMEOUT_SECONDS)
            return
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass

        try:
            self._process.kill()
        except (OSError, ValueError):
            pass
        try:
            self._process.wait(timeout=self._FORCE_STOP_TIMEOUT_SECONDS)
        except (subprocess.TimeoutExpired, OSError, ValueError):
            pass

    def _close_stdin_until(self, deadline):
        """Close the buffered pipe without letting flush block release()."""
        errors = []

        def close_stdin():
            try:
                self._process.stdin.close()
            except BaseException as error:
                errors.append(error)

        close_thread = threading.Thread(
            target=close_stdin,
            name='ffmpeg-stdin-closer',
            daemon=True,
        )
        close_thread.start()
        close_thread.join(timeout=self._remaining_seconds(deadline))
        return close_thread, errors

    def write(self, frame):
        """写入一帧（numpy BGR 数组）。"""
        with self._enqueue_lock:
            if self._released:
                raise RuntimeError('Cannot write to a released FFmpeg video writer')
            self._raise_writer_error()

            if frame.dtype != np.uint8:
                frame = np.clip(frame, 0, 255).astype(np.uint8)
            # Materialize the bytes before returning. Callers commonly reuse
            # the numpy buffer for the next frame, so queueing the array itself
            # could let later mutations corrupt an earlier frame. Doing this
            # under the producer lock also bounds memory for concurrent calls.
            frame_bytes = frame.tobytes()
            timeout_error = TimeoutError(
                f'Timed out after {self._WRITE_TIMEOUT_SECONDS:g}s waiting '
                f'for FFmpeg to accept a video frame: {self._output_path}'
            )
            deadline = time.monotonic() + self._WRITE_TIMEOUT_SECONDS
            try:
                self._put_until(frame_bytes, deadline, timeout_error)
            except TimeoutError as error:
                self._set_writer_error(error)
                self._force_stop_process()
                self._writer_thread.join(
                    timeout=self._FORCE_STOP_TIMEOUT_SECONDS,
                )
                raise

    def release(self):
        """关闭管道并等待编码完成。"""
        # A second concurrent caller must wait for the first finalization,
        # rather than returning while FFmpeg still owns the output file.
        with self._release_lock:
            if self._release_complete:
                if self._release_error is not None:
                    raise self._release_error
                return
            try:
                finalize_timeout = self._FINALIZE_TIMEOUT_SECONDS
                deadline = time.monotonic() + finalize_timeout
                timeout_error = TimeoutError(
                    f'Timed out after {finalize_timeout:g}s finalizing '
                    f'FFmpeg video writer: {self._output_path}'
                )
                process_was_forced = False
                # Publish the release intent before waiting for a producer
                # already holding the enqueue lock. Acquiring that lock is
                # itself bounded by the same finalization deadline.
                self._released = True
                remaining = self._remaining_seconds(deadline)
                enqueue_acquired = (
                    remaining > 0
                    and self._enqueue_lock.acquire(timeout=remaining)
                )
                if not enqueue_acquired:
                    self._set_writer_error(timeout_error)
                    self._force_stop_process()
                    process_was_forced = True
                else:
                    try:
                        if not self._writer_error_event.is_set():
                            try:
                                self._put_until(
                                    self._STOP,
                                    deadline,
                                    timeout_error,
                                )
                            except TimeoutError as error:
                                self._set_writer_error(error)
                                self._force_stop_process()
                                process_was_forced = True
                            except BaseException as error:
                                # The worker can fail while release is waiting
                                # for a full queue. Cleanup must still run.
                                self._set_writer_error(error)
                    finally:
                        self._enqueue_lock.release()

                # The sentinel is ordered behind every accepted frame, so
                # joining guarantees all queued frames reached FFmpeg.
                self._writer_thread.join(
                    timeout=self._remaining_seconds(deadline),
                )
                if self._writer_thread.is_alive():
                    self._set_writer_error(timeout_error)
                    if not process_was_forced:
                        self._force_stop_process()
                        process_was_forced = True
                    self._writer_thread.join(
                        timeout=self._FORCE_STOP_TIMEOUT_SECONDS,
                    )

                close_error = None
                # Closing a buffered pipe while another thread is inside
                # write() can itself block on the stream lock. Only close once
                # the worker has stopped; the process was already killed when
                # the bounded join above failed.
                if not self._writer_thread.is_alive():
                    close_thread, close_errors = self._close_stdin_until(
                        deadline,
                    )
                    if close_thread.is_alive():
                        self._set_writer_error(timeout_error)
                        if not process_was_forced:
                            self._force_stop_process()
                            process_was_forced = True
                        close_thread.join(
                            timeout=self._FORCE_STOP_TIMEOUT_SECONDS,
                        )
                    if close_errors:
                        close_error = close_errors[0]

                returncode = None
                if process_was_forced:
                    wait_error = None
                else:
                    remaining = self._remaining_seconds(deadline)
                    if remaining <= 0:
                        wait_error = timeout_error
                        self._set_writer_error(timeout_error)
                        self._force_stop_process()
                    else:
                        try:
                            returncode = self._process.wait(timeout=remaining)
                        except subprocess.TimeoutExpired:
                            wait_error = timeout_error
                            self._set_writer_error(timeout_error)
                            self._force_stop_process()
                        else:
                            wait_error = None

                if self._writer_error_event.is_set():
                    self._raise_writer_error()
                if wait_error is not None:
                    raise wait_error
                if returncode not in (None, 0):
                    raise RuntimeError(
                        f'FFmpeg video writer exited with code {returncode}: '
                        f'{self._output_path}'
                    )
                if close_error is not None:
                    raise close_error
            except BaseException as error:
                self._release_error = error
                raise
            finally:
                self._release_complete = True
