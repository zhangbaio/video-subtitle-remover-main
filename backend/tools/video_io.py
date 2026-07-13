import os
import queue
import subprocess
import tempfile
import threading
from functools import lru_cache

import cv2
import numpy as np

from .ffmpeg_cli import FFmpegCLI
from .hardware_accelerator import HardwareAccelerator
from .common_tools import get_readable_path


class FramePrefetcher:
    """
    后台线程预解码视频帧，使 I/O 与模型推理重叠。
    接口兼容 cv2.VideoCapture（read/release）。
    """

    def __init__(self, video_cap, buffer_size=24, read_timeout=30.0):
        self.cap = video_cap
        self._buffer = queue.Queue(maxsize=buffer_size)
        self._stopped = False
        self._read_timeout = max(1.0, float(read_timeout))
        self._reader_error = None
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def _read_loop(self):
        try:
            while not self._stopped:
                ret, frame = self.cap.read()
                self._buffer.put((ret, frame))
                if not ret:
                    break
        except BaseException as error:
            self._reader_error = error
            # Wake a consumer already blocked in read(). If the buffer is
            # full, queued frames are consumed before read() sees the error.
            try:
                self._buffer.put_nowait((False, None))
            except queue.Full:
                pass

    def read(self):
        """读取下一帧，接口与 cv2.VideoCapture.read() 一致。"""
        if self._reader_error is not None:
            raise RuntimeError("Video frame prefetch failed") from self._reader_error
        try:
            ret, frame = self._buffer.get(timeout=self._read_timeout)
        except queue.Empty as error:
            if self._reader_error is not None:
                raise RuntimeError("Video frame prefetch failed") from self._reader_error
            if not self._thread.is_alive():
                raise RuntimeError("Video frame prefetch stopped before end of stream") from error
            raise TimeoutError(
                f"Timed out after {self._read_timeout:g}s waiting for the next video frame"
            ) from error
        if not ret and self._reader_error is not None:
            raise RuntimeError("Video frame prefetch failed") from self._reader_error
        return ret, frame

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


class FFmpegVideoReader:
    """
    使用 FFmpeg 顺序解码视频帧；优先尝试 CUDA 硬解码，失败时由调用方回退。
    接口尽量兼容 cv2.VideoCapture（read/get/release）。
    """

    @staticmethod
    @lru_cache(maxsize=4)
    def _supports_hwaccel(ffmpeg_path, hwaccel_name):
        try:
            out = subprocess.check_output(
                [ffmpeg_path, "-hide_banner", "-hwaccels"],
                stderr=subprocess.STDOUT,
                text=True,
                errors="ignore",
            )
            return any(hwaccel_name == line.strip() for line in out.splitlines())
        except Exception:
            return False

    def __init__(self, video_path, width, height, fps=0.0, frame_count=0, prefer_cuda=True):
        self.video_path = video_path
        self.width = int(width)
        self.height = int(height)
        self.fps = float(fps or 0.0)
        self.frame_count = int(frame_count or 0)
        self._frame_bytes = self.width * self.height * 3
        self._released = False
        self._pending_frame = None
        self._ffmpeg_path = FFmpegCLI.instance().ffmpeg_path
        self._process = self._start_process(prefer_cuda=prefer_cuda)
        ret, frame = self._read_frame_bytes()
        if not ret:
            self.release()
            raise RuntimeError(f"Failed to open ffmpeg video reader: {video_path}")
        self._pending_frame = frame

    def _start_process(self, prefer_cuda=True):
        input_path = get_readable_path(self.video_path)
        cmd = [
            self._ffmpeg_path,
            "-hide_banner",
            "-loglevel", "error",
        ]
        if prefer_cuda and HardwareAccelerator.instance().has_cuda() and self._supports_hwaccel(self._ffmpeg_path, "cuda"):
            cmd.extend([
                "-hwaccel", "cuda",
                "-hwaccel_output_format", "cuda",
            ])
        cmd.extend([
            "-i", input_path,
            "-vf", "hwdownload,format=nv12,format=bgr24" if prefer_cuda and HardwareAccelerator.instance().has_cuda() and self._supports_hwaccel(self._ffmpeg_path, "cuda") else "format=bgr24",
            "-pix_fmt", "bgr24",
            "-f", "rawvideo",
            "-vsync", "0",
            "-",
        ])
        return subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
            bufsize=self._frame_bytes * 2,
        )

    def _read_frame_bytes(self):
        if self._released or self._process.stdout is None:
            return False, None
        raw = self._process.stdout.read(self._frame_bytes)
        if len(raw) != self._frame_bytes:
            return False, None
        frame = np.frombuffer(raw, dtype=np.uint8).reshape((self.height, self.width, 3)).copy()
        return True, frame

    def read(self):
        if self._pending_frame is not None:
            frame = self._pending_frame
            self._pending_frame = None
            return True, frame
        return self._read_frame_bytes()

    def get(self, propId):
        if propId == cv2.CAP_PROP_FRAME_WIDTH:
            return self.width
        if propId == cv2.CAP_PROP_FRAME_HEIGHT:
            return self.height
        if propId == cv2.CAP_PROP_FPS:
            return self.fps
        if propId == cv2.CAP_PROP_FRAME_COUNT:
            return self.frame_count
        return 0

    def release(self):
        if self._released:
            return
        self._released = True
        if self._process.stdout is not None:
            try:
                self._process.stdout.close()
            except OSError:
                pass
        if self._process.poll() is None:
            self._process.terminate()
            try:
                self._process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self._process.kill()
                self._process.wait(timeout=5)


def create_processing_capture(
    video_path,
    width,
    height,
    fps=0.0,
    frame_count=0,
    fallback_cap=None,
    prefer_cuda=None,
):
    if prefer_cuda is None:
        prefer_cuda = HardwareAccelerator.instance().has_cuda()
    if prefer_cuda:
        try:
            return FFmpegVideoReader(video_path, width, height, fps=fps, frame_count=frame_count, prefer_cuda=True)
        except Exception:
            pass
    if fallback_cap is not None:
        return fallback_cap
    cap = cv2.VideoCapture(get_readable_path(video_path))
    return cap


class FFmpegVideoWriter:
    """
    通过 FFmpeg 管道写入帧，使用 libx264 编码。
    接口兼容 cv2.VideoWriter（write/release）。
    """

    @staticmethod
    @lru_cache(maxsize=4)
    def _supports_encoder(ffmpeg_path, encoder_name):
        try:
            out = subprocess.check_output(
                [ffmpeg_path, "-hide_banner", "-encoders"],
                stderr=subprocess.STDOUT,
                text=True,
                errors="ignore",
            )
            return any(encoder_name in line for line in out.splitlines())
        except Exception:
            return False

    def __init__(self, output_path, fps, size, bitrate_mbps=4.5):
        w, h = size
        bitrate_mbps = max(0.1, float(bitrate_mbps))
        bitrate = f'{bitrate_mbps:g}M'
        ffmpeg_path = FFmpegCLI.instance().ffmpeg_path
        prefer_nvenc = (
            HardwareAccelerator.instance().has_cuda()
            and self._supports_encoder(ffmpeg_path, "h264_nvenc")
        )

        cmd = [
            ffmpeg_path,
            '-y',
            '-f', 'rawvideo',
            '-vcodec', 'rawvideo',
            '-s', f'{w}x{h}',
            '-pix_fmt', 'bgr24',
            '-r', str(fps),
            '-i', '-',
        ]

        if prefer_nvenc:
            cmd.extend([
                '-c:v', 'h264_nvenc',
                '-preset', 'p4',
                '-rc', 'cbr',
                '-b:v', bitrate,
                '-minrate', bitrate,
                '-maxrate', bitrate,
                '-bufsize', bitrate,
                '-pix_fmt', 'yuv420p',
            ])
        else:
            cmd.extend([
                '-c:v', 'libx264',
                '-preset', 'fast',
                '-b:v', bitrate,
                '-minrate', bitrate,
                '-maxrate', bitrate,
                '-bufsize', bitrate,
                '-x264-params', 'nal-hrd=cbr:force-cfr=1',
                '-pix_fmt', 'yuv420p',
            ])

        cmd.extend([
            '-loglevel', 'error',
            output_path
        ])
        self._output_path = output_path
        self._frame_shape = (int(h), int(w), 3)
        self._released = False
        # A real file cannot fill up and block FFmpeg like an unread stderr
        # pipe, but still preserves diagnostics if the encoder fails.
        self._stderr_file = tempfile.TemporaryFile(mode='w+b')
        try:
            self._process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=self._stderr_file,
            )
        except BaseException:
            self._stderr_file.close()
            raise

    def _read_stderr(self, limit=16384):
        if self._stderr_file.closed:
            return ''
        try:
            self._stderr_file.flush()
            self._stderr_file.seek(0, os.SEEK_END)
            size = self._stderr_file.tell()
            offset = max(0, size - limit)
            self._stderr_file.seek(offset)
            message = self._stderr_file.read(limit).decode('utf-8', errors='replace').strip()
            if offset:
                message = f'...{message}'
            return message
        except OSError:
            return ''

    def _ffmpeg_error(self, action):
        returncode = self._process.poll()
        status = '' if returncode is None else f' (exit code {returncode})'
        message = f'FFmpeg video writer {action}{status}: {self._output_path}'
        stderr = self._read_stderr()
        if stderr:
            message = f'{message}\n{stderr}'
        return RuntimeError(message)

    def write(self, frame):
        """写入一帧（numpy BGR 数组）。"""
        if self._released:
            raise RuntimeError('Cannot write to a released FFmpeg video writer')
        if frame.shape != self._frame_shape:
            raise ValueError(
                f'Expected a BGR frame with shape {self._frame_shape}, got {frame.shape}'
            )
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        if not frame.flags.c_contiguous:
            frame = np.ascontiguousarray(frame)

        if self._process.poll() is not None:
            raise self._ffmpeg_error('exited before accepting a frame')
        stdin = self._process.stdin
        if stdin is None or stdin.closed:
            raise self._ffmpeg_error('has no writable input pipe')

        # Passing the contiguous buffer directly avoids frame.tobytes(), which
        # otherwise allocates and copies the complete frame before every write.
        remaining = memoryview(frame).cast('B')
        try:
            while remaining:
                written = stdin.write(remaining)
                if not written:
                    raise BrokenPipeError('FFmpeg input pipe accepted zero bytes')
                remaining = remaining[written:]
        except (BrokenPipeError, OSError, ValueError) as error:
            try:
                self._process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                pass
            raise self._ffmpeg_error('failed while writing a frame') from error

    def release(self):
        """关闭管道并等待编码完成。"""
        if self._released:
            return
        self._released = True
        close_error = None
        try:
            stdin = self._process.stdin
            if stdin is not None and not stdin.closed:
                try:
                    stdin.close()
                except (BrokenPipeError, OSError, ValueError) as error:
                    close_error = error

            try:
                returncode = self._process.wait(timeout=600)
            except subprocess.TimeoutExpired as error:
                self._process.terminate()
                try:
                    self._process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self._process.kill()
                    self._process.wait(timeout=5)
                raise self._ffmpeg_error('timed out while finalizing') from error

            if returncode != 0:
                raise self._ffmpeg_error('failed while finalizing')
            if close_error is not None:
                raise self._ffmpeg_error('failed to close its input pipe') from close_error
        finally:
            self._stderr_file.close()
