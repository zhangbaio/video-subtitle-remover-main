import unittest
from unittest import mock

import numpy as np

from backend.main import SubtitleRemover
from ui.home_interface import (
    _normalized_areas_to_video,
    _sanitize_normalized_video_areas,
    _video_areas_to_normalized,
)


class FixedWatermarkCoordinateTests(unittest.TestCase):
    def test_video_normalized_round_trip(self):
        areas = [(14, 76, 982, 1270)]

        normalized = _video_areas_to_normalized(areas, 1280, 720)
        restored = _normalized_areas_to_video(normalized, 1280, 720)

        self.assertEqual(restored, areas)

    def test_normalized_areas_are_clamped_and_invalid_removed(self):
        areas = _sanitize_normalized_video_areas([
            (-0.1, 0.2, 0.8, 1.2),
            (0.5, 0.5, 0.1, 0.2),
            ("bad", 0.5, 0.1, 0.2),
        ])

        self.assertEqual(areas, [(0.0, 0.2, 0.8, 1.0)])


class _FakeReader:
    def __init__(self, _capture, frames):
        self.frames = iter(frames)
        self.stopped = False

    def read(self):
        try:
            return True, next(self.frames)
        except StopIteration:
            return False, None

    def stop(self):
        self.stopped = True


class _FakeWriter:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())


class _FakeLocalInpaint:
    instances = []

    def __init__(self, *_args, **_kwargs):
        self.calls = []
        self.__class__.instances.append(self)

    def __call__(self, frames, _mask, active_frames):
        self.calls.append((list(frames), list(active_frames)))
        return [
            frame + 1 if active else frame.copy()
            for frame, active in zip(frames, active_frames)
        ]


class FixedWatermarkStreamingTests(unittest.TestCase):
    def _make_remover(self, frames, ab_sections=None):
        remover = object.__new__(SubtitleRemover)
        remover.mask_size = frames[0].shape[:2]
        remover.sub_areas = [(1, 3, 1, 3)]
        remover.sttn_det_inpaint = object()
        remover.video_cap = object()
        remover.video_writer = _FakeWriter()
        remover.ab_sections = ab_sections
        remover.gui_mode = False
        remover.preview_emit_interval = 0
        remover._last_preview_emit_time = 0
        remover.append_output = mock.Mock()
        remover.push_preview_with_comp = mock.Mock()
        remover.update_progress = mock.Mock()
        return remover

    def test_streaming_reprocesses_overlap_without_duplicate_output(self):
        frames = [
            np.full((4, 4, 3), index, dtype=np.uint8)
            for index in range(55)
        ]
        remover = self._make_remover(frames)
        reader = _FakeReader(None, frames)
        _FakeLocalInpaint.instances.clear()

        with (
            mock.patch("backend.main.FramePrefetcher", return_value=reader),
            mock.patch("backend.main.FixedWatermarkInpaint", _FakeLocalInpaint),
        ):
            remover.fixed_watermark_mode(mock.Mock())

        model = _FakeLocalInpaint.instances[0]
        self.assertEqual([len(call[0]) for call in model.calls], [50, 15])
        self.assertEqual(len(remover.video_writer.frames), 55)
        for index, output in enumerate(remover.video_writer.frames):
            np.testing.assert_array_equal(output, frames[index] + 1)
        self.assertTrue(reader.stopped)

    def test_ab_sections_leave_inactive_frames_unchanged(self):
        frames = [
            np.full((4, 4, 3), index, dtype=np.uint8)
            for index in range(6)
        ]
        remover = self._make_remover(frames, ab_sections=[range(2, 4)])
        reader = _FakeReader(None, frames)
        _FakeLocalInpaint.instances.clear()

        with (
            mock.patch("backend.main.FramePrefetcher", return_value=reader),
            mock.patch("backend.main.FixedWatermarkInpaint", _FakeLocalInpaint),
        ):
            remover.fixed_watermark_mode(mock.Mock())

        for index, output in enumerate(remover.video_writer.frames):
            expected = frames[index] + 1 if index in (2, 3) else frames[index]
            np.testing.assert_array_equal(output, expected)


if __name__ == "__main__":
    unittest.main()
