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
