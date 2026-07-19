import threading
import types
import unittest
from unittest import mock

import cv2
import numpy as np

from backend.main import SubtitleRemover, _map_stage_progress
from backend.tools import subtitle_detect
from backend.tools.subtitle_detect import SubtitleDetect


class _FakeCapture:
    def __init__(self, frame_count):
        self.frames = [
            np.zeros((8, 8, 3), dtype=np.uint8)
            for _ in range(frame_count)
        ]
        self.index = 0

    @staticmethod
    def isOpened():
        return True

    def get(self, property_id):
        if property_id == cv2.CAP_PROP_FRAME_COUNT:
            return len(self.frames)
        return 0

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame

    def release(self):
        pass


class _FakeProgressBar:
    def __init__(self, total=0, **_kwargs):
        self.total = total
        self.n = 0

    def update(self, increment):
        self.n += increment


class SttnDetectionProgressTests(unittest.TestCase):
    def test_detection_reports_every_decoded_frame_including_skipped_sections(self):
        detector = object.__new__(SubtitleDetect)
        detector.video_path = "video.mp4"
        detector.sub_areas = []
        detector.SAMPLE_STEP = 3
        detector.detect_subtitle = mock.Mock(return_value=[])
        remover = types.SimpleNamespace(
            ab_sections=[range(0, 1)],
            append_output=mock.Mock(),
            progress_total=0,
        )
        progress = []

        with (
            mock.patch.object(
                subtitle_detect.cv2,
                "VideoCapture",
                return_value=_FakeCapture(4),
            ),
            mock.patch.object(
                subtitle_detect,
                "safe_tqdm",
                side_effect=lambda **kwargs: _FakeProgressBar(**kwargs),
            ),
        ):
            result = detector.find_subtitle_frame_no(
                sub_remover=remover,
                progress_callback=lambda completed, total: progress.append(
                    (completed, total)
                ),
            )

        self.assertEqual(result, {})
        self.assertEqual(
            progress,
            [(0, 4), (1, 4), (2, 4), (3, 4), (4, 4)],
        )
        detector.detect_subtitle.assert_called_once()

    def test_roi_detection_flushes_the_final_partial_batch_before_full_progress(self):
        detector = object.__new__(SubtitleDetect)
        detector.video_path = "video.mp4"
        detector.sub_areas = [(6, 8, 0, 8)]
        detector.SAMPLE_STEP = 3
        events = []

        def detect_batch(frames):
            events.append(("ocr", len(frames)))
            return [[(1, 4, 2, 5)] for _frame in frames]

        detector.detect_subtitle_batch = mock.Mock(side_effect=detect_batch)
        progress = []

        with (
            mock.patch.object(
                subtitle_detect.cv2,
                "VideoCapture",
                return_value=_FakeCapture(8),
            ),
            mock.patch.object(
                subtitle_detect,
                "safe_tqdm",
                side_effect=lambda **kwargs: _FakeProgressBar(**kwargs),
            ),
        ):
            timeline = detector.find_subtitle_frame_no(
                progress_callback=lambda completed, total: (
                    progress.append((completed, total)),
                    events.append(("progress", completed)),
                ),
            )

        detector.detect_subtitle_batch.assert_called_once()
        self.assertEqual(events.count(("ocr", 3)), 1)
        self.assertLess(events.index(("ocr", 3)), events.index(("progress", 8)))
        self.assertEqual(progress, [(frame_no, 8) for frame_no in range(9)])
        self.assertEqual(sorted(timeline), list(range(1, 8)))

    def test_roi_batch_only_contains_sampled_frames_inside_ab_sections(self):
        detector = object.__new__(SubtitleDetect)
        detector.video_path = "video.mp4"
        detector.sub_areas = [(6, 8, 0, 8)]
        detector.SAMPLE_STEP = 3
        detector.detect_subtitle_batch = mock.Mock(
            return_value=[[(1, 4, 2, 5)], [(2, 5, 2, 5)]]
        )
        remover = types.SimpleNamespace(
            ab_sections=[range(3, 7)],
            append_output=mock.Mock(),
            progress_total=0,
        )

        with (
            mock.patch.object(
                subtitle_detect.cv2,
                "VideoCapture",
                return_value=_FakeCapture(10),
            ),
            mock.patch.object(
                subtitle_detect,
                "safe_tqdm",
                side_effect=lambda **kwargs: _FakeProgressBar(**kwargs),
            ),
        ):
            timeline = detector.find_subtitle_frame_no(sub_remover=remover)

        frames = detector.detect_subtitle_batch.call_args.args[0]
        self.assertEqual(len(frames), 2)
        self.assertEqual(sorted(timeline), [4, 5, 6, 7])

    def test_cancellation_discards_a_pending_roi_batch(self):
        detector = object.__new__(SubtitleDetect)
        detector.video_path = "video.mp4"
        detector.sub_areas = [(6, 8, 0, 8)]
        detector.SAMPLE_STEP = 3
        detector.detect_subtitle_batch = mock.Mock()
        cancel_event = threading.Event()

        def cancel_after_first_frame(completed, _total):
            if completed == 1:
                cancel_event.set()

        with (
            mock.patch.object(
                subtitle_detect.cv2,
                "VideoCapture",
                return_value=_FakeCapture(8),
            ),
            mock.patch.object(
                subtitle_detect,
                "safe_tqdm",
                side_effect=lambda **kwargs: _FakeProgressBar(**kwargs),
            ),
            self.assertRaises(subtitle_detect.SubtitleDetectionCancelled),
        ):
            detector.find_subtitle_frame_no(
                progress_callback=cancel_after_first_frame,
                cancel_event=cancel_event,
            )

        detector.detect_subtitle_batch.assert_not_called()

    def test_progress_mapping_uses_two_equal_stages(self):
        self.assertEqual(_map_stage_progress(0, 100, 0, 50), 0)
        self.assertEqual(_map_stage_progress(50, 100, 0, 50), 25)
        self.assertEqual(_map_stage_progress(100, 100, 0, 50), 50)
        self.assertEqual(_map_stage_progress(0, 100, 50, 100), 50)
        self.assertEqual(_map_stage_progress(50, 100, 50, 100), 75)
        self.assertEqual(_map_stage_progress(100, 100, 50, 100), 100)

    def test_default_removal_progress_remains_zero_to_one_hundred(self):
        remover = object.__new__(SubtitleRemover)
        remover.progress_total = 0
        remover.progress_remover = 0
        remover.notify_progress_listeners = mock.Mock()
        progress_bar = _FakeProgressBar(total=4)

        remover.update_progress(progress_bar, increment=1)

        self.assertEqual(remover.progress_total, 25)
        remover.notify_progress_listeners.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
