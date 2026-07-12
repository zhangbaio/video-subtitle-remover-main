import os
import queue
import subprocess
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

    def __init__(self, video_cap, buffer_size=24):
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


def create_processing_capture(video_path, width, height, fps=0.0, frame_count=0, fallback_cap=None):
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

    def __init__(self, output_path, fps, size):
        w, h = size
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
                '-rc', 'vbr',
                '-cq', '19',
                '-b:v', '0',
                '-pix_fmt', 'yuv420p',
            ])
        else:
            cmd.extend([
                '-c:v', 'libx264',
                '-pix_fmt', 'yuv420p',
                '-crf', '18',
                '-preset', 'fast',
            ])

        cmd.extend([
            '-loglevel', 'error',
            output_path
        ])
        self._process = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

    def write(self, frame):
        """写入一帧（numpy BGR 数组）。"""
        if frame.dtype != np.uint8:
            frame = np.clip(frame, 0, 255).astype(np.uint8)
        try:
            self._process.stdin.write(frame.tobytes())
        except BrokenPipeError:
            pass

    def release(self):
        """关闭管道并等待编码完成。"""
        try:
            self._process.stdin.close()
        except BrokenPipeError:
            pass
        try:
            self._process.wait(timeout=600)
        except subprocess.TimeoutExpired:
            self._process.terminate()
            self._process.wait(timeout=5)
