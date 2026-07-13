import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest import mock

import numpy as np
import scipy.ndimage
import torch

WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from backend import main as backend_main
from backend.main import SubtitleRemover, _preferred_audio_stream_spec
from backend.inpaint.propainter_inpaint import (
    _binary_dilate,
    _stack_uint8_images_to_tensor,
    read_mask,
)
from backend.tools.video_io import FFmpegVideoWriter


class _FakeProgressBar:
    def __init__(self, total):
        self.n = 0
        self.total = total

    def update(self, increment):
        self.n += increment


class _FakePropainter:
    def __init__(self):
        self.raft_iterations = []

    def inpaint_fixed_watermark(self, frames, mask, **kwargs):
        self.raft_iterations.append(kwargs.get("raft_iter"))
        return list(frames)


class _ShortWritePipe:
    def __init__(self, chunk_size=7):
        self.chunk_size = chunk_size
        self.closed = False
        self.parts = []

    def write(self, data):
        self.asserted_type = type(data)
        size = min(self.chunk_size, len(data))
        self.parts.append(bytes(data[:size]))
        return size


class _RunningProcess:
    def __init__(self, pipe):
        self.stdin = pipe

    @staticmethod
    def poll():
        return None


class SafeOptimizationTests(unittest.TestCase):
    def test_preview_builder_is_lazy_and_respects_force(self):
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.gui_mode = True
        remover.preview_emit_interval = 3600.0
        remover._last_preview_emit_time = float("inf")
        received = []
        remover.update_preview_with_comp = lambda left, right: received.append((left, right))
        calls = []

        def build_preview():
            calls.append(True)
            return "preview"

        remover.push_preview_with_comp(build_preview, "completed")
        self.assertEqual(calls, [])
        self.assertEqual(received, [])

        remover.push_preview_with_comp(build_preview, "completed", force=True)
        self.assertEqual(calls, [True])
        self.assertEqual(received, [("preview", "completed")])

    def test_fast_mode_is_taken_from_job_instance(self):
        remover = SubtitleRemover.__new__(SubtitleRemover)
        fake_model = _FakePropainter()
        remover.__dict__["propainter_inpaint_model"] = fake_model
        frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(2)]
        mask = np.zeros((8, 8), dtype=np.uint8)

        remover.moving_watermark_fast_mode = True
        remover._inpaint_fixed_watermark_batch(frames, mask, fast=True)
        remover.moving_watermark_fast_mode = False
        remover._inpaint_fixed_watermark_batch(frames, mask, fast=True)

        self.assertEqual(fake_model.raft_iterations, [8, None])

    def test_propainter_weights_are_reused_when_batch_length_changes(self):
        cache_backup = dict(backend_main._PROPAINTER_INPAINT_CACHE)
        backend_main._PROPAINTER_INPAINT_CACHE.clear()
        fake_model = SimpleNamespace(sub_video_length=80)
        cache_key = ("cpu:None", "model-dir")
        backend_main._PROPAINTER_INPAINT_CACHE[cache_key] = fake_model
        try:
            remover = SubtitleRemover.__new__(SubtitleRemover)
            remover.hardware_accelerator = SimpleNamespace(
                has_cuda=lambda: False,
                device=torch.device("cpu"),
            )
            remover.model_config = SimpleNamespace(PROPAINTER_MODEL_DIR="model-dir")
            remover.propainter_max_load_num = 37

            self.assertIs(remover.propainter_inpaint_model, fake_model)
            self.assertEqual(fake_model.sub_video_length, 37)
            self.assertEqual(len(backend_main._PROPAINTER_INPAINT_CACHE), 1)
        finally:
            backend_main._PROPAINTER_INPAINT_CACHE.clear()
            backend_main._PROPAINTER_INPAINT_CACHE.update(cache_backup)

    def test_progress_notifications_are_deduplicated(self):
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.progress_total = 0
        remover.progress_remover = 0
        notifications = []
        remover.notify_progress_listeners = lambda: notifications.append(remover.progress_total)
        progress = _FakeProgressBar(total=1000)

        remover.update_progress(progress, 1)
        remover.update_progress(progress, 1)
        self.assertEqual(notifications, [])

        remover.update_progress(progress, 8)
        self.assertEqual(notifications, [1])
        remover.update_progress(progress, 1)
        self.assertEqual(notifications, [1])

    def test_audio_mux_uses_one_ffmpeg_invocation(self):
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.video_path = "source.mp4"
        remover.video_out_path = "output.mp4"
        remover.is_successful_merged = False
        remover.append_output = mock.Mock()

        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_video:
            temp_path = temp_video.name
        try:
            remover.video_temp_file = mock.Mock()
            remover.video_temp_file.name = temp_path
            with mock.patch("backend.main.subprocess.check_output") as check_output:
                remover.merge_audio_to_video()

            check_output.assert_called_once()
            command = check_output.call_args.args[0]
            self.assertIn("0:v:0", command)
            self.assertIn("1:a:0?", command)
            self.assertEqual(command.count("-i"), 2)
            self.assertTrue(remover.is_successful_merged)
            remover.video_temp_file.close.assert_called_once()
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def test_audio_stream_choice_matches_ffmpeg_channel_priority(self):
        def stream(index, channels):
            return SimpleNamespace(
                index=index,
                codec_context=SimpleNamespace(
                    channels=channels,
                    layout=SimpleNamespace(channels=tuple(range(channels))),
                ),
            )

        container = mock.MagicMock()
        container.__enter__.return_value = container
        container.streams.audio = [stream(1, 2), stream(3, 6), stream(4, 6)]
        with mock.patch("av.open", return_value=container):
            self.assertEqual(_preferred_audio_stream_spec("source.mp4"), "1:3?")

    def test_job_wrapper_releases_resources_without_masking_original_error(self):
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.isFinished = False
        remover.video_cap = mock.Mock()
        remover.video_writer = mock.Mock()
        remover.video_writer.release.side_effect = RuntimeError("cleanup failed")
        remover._run_job = mock.Mock(side_effect=ValueError("processing failed"))

        with tempfile.NamedTemporaryFile(delete=False) as temp_file:
            temp_path = temp_file.name
        remover.video_temp_file = mock.Mock()
        remover.video_temp_file.name = temp_path

        with self.assertRaisesRegex(ValueError, "processing failed"):
            remover.run()

        remover.video_cap.release.assert_called_once()
        remover.video_writer.release.assert_called_once()
        remover.video_temp_file.close.assert_called_once()
        self.assertFalse(os.path.exists(temp_path))

    def test_video_writer_handles_short_writes_in_order(self):
        writer = FFmpegVideoWriter.__new__(FFmpegVideoWriter)
        writer._released = False
        writer._frame_shape = (2, 3, 3)
        writer._output_path = "unused.mp4"
        pipe = _ShortWritePipe()
        writer._process = _RunningProcess(pipe)
        frame = np.arange(18, dtype=np.uint8).reshape(2, 3, 3)

        writer.write(frame)

        self.assertIs(pipe.asserted_type, memoryview)
        self.assertEqual(b"".join(pipe.parts), frame.tobytes())

    def test_opencv_binary_dilation_matches_scipy(self):
        rng = np.random.default_rng(7)
        for shape in ((1, 1), (3, 9), (17, 13)):
            mask = (rng.random(shape) > 0.75).astype(np.uint8) * 255
            for iterations in (1, 2, 4, 8):
                expected = scipy.ndimage.binary_dilation(
                    mask,
                    iterations=iterations,
                ).astype(np.uint8)
                np.testing.assert_array_equal(
                    _binary_dilate(mask, iterations),
                    expected,
                )

    def test_direct_tensor_stack_matches_previous_layout(self):
        rng = np.random.default_rng(11)
        frames = [rng.integers(0, 256, (5, 7, 3), dtype=np.uint8) for _ in range(4)]
        old_stack = np.stack(frames, axis=2)
        expected = torch.from_numpy(old_stack).permute(2, 3, 0, 1).contiguous().float().div(255)
        self.assertTrue(torch.equal(_stack_uint8_images_to_tensor(frames), expected))

        masks = [rng.integers(0, 256, (5, 7), dtype=np.uint8) for _ in range(4)]
        old_mask_stack = np.stack([mask[:, :, None] for mask in masks], axis=2)
        expected_masks = (
            torch.from_numpy(old_mask_stack)
            .permute(2, 3, 0, 1)
            .contiguous()
            .float()
            .div(255)
        )
        self.assertTrue(torch.equal(_stack_uint8_images_to_tensor(masks), expected_masks))

    def test_equal_mask_dilations_are_reused_without_changing_values(self):
        mask = np.zeros((13, 19), dtype=np.uint8)
        mask[4:8, 6:11] = 255
        flow_masks, masks = read_mask(
            mask,
            length=3,
            size=(19, 13),
            flow_mask_dilates=4,
            mask_dilates=4,
            as_numpy=True,
        )
        self.assertEqual(flow_masks.shape, (3, 13, 19))
        np.testing.assert_array_equal(flow_masks, masks)
        expected = scipy.ndimage.binary_dilation(mask, iterations=4).astype(np.uint8) * 255
        for index in range(3):
            np.testing.assert_array_equal(flow_masks[index], expected)


if __name__ == "__main__":
    unittest.main()
