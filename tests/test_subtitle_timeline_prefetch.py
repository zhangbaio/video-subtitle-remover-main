import os
import tempfile
import threading
import types
import unittest
from unittest import mock

import cv2
import numpy as np

from backend.tools import subtitle_detect
from backend.tools.constant import InpaintMode
from backend.tools.subtitle_detect import SubtitleDetect, SubtitleDetectionCancelled
from ui.home_interface import HomeInterface, _preview_areas_to_video_coordinates


class SubtitleTimelineCacheTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory(dir=os.getcwd())
        self.video_path = os.path.join(self.temp_dir.name, "episode.mp4")
        self.model_dir = os.path.join(self.temp_dir.name, "model")
        os.makedirs(self.model_dir)
        with open(self.video_path, "wb") as file:
            file.write(b"video")
        with open(os.path.join(self.model_dir, "inference.pdiparams"), "wb") as file:
            file.write(b"model")
        self.model_config = types.SimpleNamespace(
            DET_MODEL_NAME="PP-OCRv5_server_det",
            DET_MODEL_DIR=self.model_dir,
        )

    def tearDown(self):
        self.temp_dir.cleanup()

    def _detector(self, areas=((800, 900, 100, 600),), sample_step=3):
        detector = object.__new__(SubtitleDetect)
        detector.video_path = self.video_path
        detector.sub_areas = list(areas)
        detector.SAMPLE_STEP = sample_step
        return detector

    def test_cache_validates_path_file_area_model_and_sample_step(self):
        with mock.patch.object(subtitle_detect, "ModelConfig", return_value=self.model_config):
            detector = self._detector()
            timeline = {1: [(100, 600, 800, 900)]}
            cache = detector.create_timeline_cache(timeline)
            self.assertEqual(detector.get_cached_timeline(cache), timeline)
            relative_detector = self._detector()
            relative_detector.video_path = os.path.relpath(self.video_path)
            self.assertEqual(relative_detector.get_cached_timeline(cache), timeline)

            self.assertIsNone(self._detector(areas=((801, 900, 100, 600),)).get_cached_timeline(cache))
            self.assertIsNone(self._detector(sample_step=4).get_cached_timeline(cache))

            with open(self.video_path, "ab") as file:
                file.write(b"changed")
            self.assertIsNone(detector.get_cached_timeline(cache))

    def test_model_file_change_invalidates_cache(self):
        with mock.patch.object(subtitle_detect, "ModelConfig", return_value=self.model_config):
            detector = self._detector()
            cache = detector.create_timeline_cache({})
            with open(os.path.join(self.model_dir, "inference.pdiparams"), "ab") as file:
                file.write(b"changed")
            self.assertIsNone(detector.get_cached_timeline(cache))

    def test_malformed_timeline_is_rejected(self):
        with mock.patch.object(subtitle_detect, "ModelConfig", return_value=self.model_config):
            detector = self._detector()
            cache = detector.create_timeline_cache({"1": "bad"})
            self.assertIsNone(detector.get_cached_timeline(cache))


class SubtitleTimelinePrefetchTests(unittest.TestCase):
    class _Capture:
        def __init__(self, frame_count=4):
            self.frames = [np.zeros((8, 8, 3), dtype=np.uint8) for _ in range(frame_count)]
            self.index = 0
            self.released = False

        def isOpened(self):
            return not self.released

        def get(self, prop):
            if prop == cv2.CAP_PROP_FRAME_COUNT:
                return len(self.frames)
            return 0

        def read(self):
            if self.index >= len(self.frames):
                return False, None
            frame = self.frames[self.index]
            self.index += 1
            return True, frame

        def release(self):
            self.released = True

    class _Progress:
        def __init__(self, total=0, **_kwargs):
            self.total = total
            self.n = 0
            self.closed = False

        def update(self, value):
            self.n += value

        def close(self):
            self.closed = True

    def test_preview_conversion_matches_vertical_video_geometry(self):
        frame_width, frame_height = 720, 1280
        preview_width, preview_height = 960, 540
        scaled_width = 303 / preview_width
        border_left = 328 / preview_width
        video_area = (800, 900, 100, 600)
        preview_area = (
            video_area[0] / frame_height,
            video_area[1] / frame_height,
            border_left + video_area[2] / frame_width * scaled_width,
            border_left + video_area[3] / frame_width * scaled_width,
        )

        converted = _preview_areas_to_video_coordinates(
            [preview_area],
            frame_width,
            frame_height,
            preview_width,
            preview_height,
        )

        self.assertEqual(converted, [video_area])

    def test_wait_for_future_stops_on_cancel(self):
        cancel_event = threading.Event()
        cancel_event.set()
        future = mock.Mock()

        with self.assertRaises(SubtitleDetectionCancelled):
            HomeInterface._wait_for_future_or_cancel(future, cancel_event)
        future.result.assert_not_called()

    def test_cancel_does_not_hold_lock_while_cancelling_future(self):
        lock = threading.Lock()

        class _Future:
            def __init__(self):
                self.cancelled_without_lock = False

            def cancel(inner_self):
                inner_self.cancelled_without_lock = lock.acquire(blocking=False)
                if inner_self.cancelled_without_lock:
                    lock.release()

        future = _Future()
        fake_home = types.SimpleNamespace(
            _subtitle_timeline_cancel_event=threading.Event(),
            _subtitle_timeline_lock=lock,
            _subtitle_timeline_futures={"episode": future},
            _subtitle_timeline_results={"episode": {}},
            _subtitle_timeline_errors={"episode": RuntimeError("ignored")},
        )

        HomeInterface._cancel_subtitle_timeline_prefetch(fake_home)

        self.assertTrue(future.cancelled_without_lock)
        self.assertFalse(fake_home._subtitle_timeline_futures)
        self.assertFalse(fake_home._subtitle_timeline_results)
        self.assertFalse(fake_home._subtitle_timeline_errors)

    def test_timeline_wait_timeout_cancels_prefetch_and_falls_back(self):
        future = mock.Mock()
        log_signal = types.SimpleNamespace(emit=mock.Mock())
        fake_home = types.SimpleNamespace(
            SUBTITLE_TIMELINE_WAIT_TIMEOUT_SECONDS=0.0,
            _stop_event=threading.Event(),
            _subtitle_timeline_lock=threading.Lock(),
            _subtitle_timeline_results={},
            _subtitle_timeline_futures={"episode": future},
            _subtitle_timeline_errors={},
            _subtitle_timeline_key=lambda _path: "episode",
            append_log_signal=log_signal,
            _reset_subtitle_timeline_prefetch=mock.Mock(),
        )

        cache = HomeInterface._take_subtitle_timeline_cache(fake_home, "episode.mp4")

        self.assertIsNone(cache)
        fake_home._reset_subtitle_timeline_prefetch.assert_called_once_with()
        future.result.assert_not_called()
        messages = [message for call in log_signal.emit.call_args_list for message in call.args[0]]
        self.assertTrue(any("回退当前任务正常检测" in message for message in messages))

    def test_complete_scan_does_not_require_a_sub_remover(self):
        detector = object.__new__(SubtitleDetect)
        detector.video_path = "episode.mp4"
        detector.sub_areas = []
        detector.SAMPLE_STEP = 3
        detector.detect_subtitle = mock.Mock(return_value=[(1, 4, 2, 5)])

        with (
            mock.patch.object(subtitle_detect.cv2, "VideoCapture", return_value=self._Capture()),
            mock.patch.object(
                subtitle_detect,
                "safe_tqdm",
                side_effect=lambda **kwargs: self._Progress(**kwargs),
            ),
        ):
            timeline = detector.find_subtitle_frame_no()

        self.assertEqual(sorted(timeline), [1, 2, 3, 4])
        self.assertEqual(detector.detect_subtitle.call_count, 2)

    def test_complete_scan_honors_cancellation(self):
        detector = object.__new__(SubtitleDetect)
        detector.video_path = "episode.mp4"
        detector.sub_areas = []
        detector.SAMPLE_STEP = 3
        cancel_event = threading.Event()
        cancel_event.set()
        capture = self._Capture()

        progress = self._Progress()
        with (
            mock.patch.object(subtitle_detect.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(subtitle_detect, "safe_tqdm", return_value=progress),
            self.assertRaises(SubtitleDetectionCancelled),
        ):
            detector.find_subtitle_frame_no(cancel_event=cancel_event)

        self.assertTrue(capture.released)
        self.assertTrue(progress.closed)

    def test_complete_scan_releases_resources_when_detector_raises(self):
        detector = object.__new__(SubtitleDetect)
        detector.video_path = "episode.mp4"
        detector.sub_areas = []
        detector.SAMPLE_STEP = 3
        detector.detect_subtitle = mock.Mock(side_effect=RuntimeError("OCR failed"))
        capture = self._Capture()
        progress = self._Progress()

        with (
            mock.patch.object(subtitle_detect.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(subtitle_detect, "safe_tqdm", return_value=progress),
            self.assertRaisesRegex(RuntimeError, "OCR failed"),
        ):
            detector.find_subtitle_frame_no()

        self.assertTrue(capture.released)
        self.assertTrue(progress.closed)

    def test_sttn_stage_schedules_full_timeline_prefetch(self):
        fake_home = types.SimpleNamespace(
            current_processing_task_index=3,
            _auto_area_prefetch_started_for_task=None,
            _resume_auto_area_preprocessing=mock.Mock(),
            _schedule_pending_auto_area_detections=mock.Mock(),
            _schedule_subtitle_timeline_prefetch=mock.Mock(),
        )
        fake_home._maybe_start_auto_area_prefetch = types.MethodType(
            HomeInterface._maybe_start_auto_area_prefetch,
            fake_home,
        )

        self.assertTrue(
            fake_home._maybe_start_auto_area_prefetch(
                50,
                inpaint_mode=InpaintMode.STTN_DET,
            )
        )
        fake_home._schedule_subtitle_timeline_prefetch.assert_called_once_with()


if __name__ == "__main__":
    unittest.main()
