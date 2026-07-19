import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np

from backend.tools import subtitle_detect
from backend.tools.subtitle_detect import SubtitleDetect, get_default_subtitle_area
from ui import home_interface
from ui.component.task_list_component import TaskOptions
from ui.home_interface import HomeInterface, _are_normalized_preview_areas


class _NonReentrantPredictor:
    def __init__(self):
        self._state_lock = threading.Lock()
        self._active = False
        self.max_active = 0

    def predict(self, _image):
        with self._state_lock:
            if self._active:
                raise RuntimeError("predictor was entered concurrently")
            self._active = True
            self.max_active = max(self.max_active, 1)
        try:
            time.sleep(0.03)
            return []
        finally:
            with self._state_lock:
                self._active = False


class _TaskListStub:
    def __init__(self):
        self.values = {
            TaskOptions.SUB_AREAS: [(0, 1280, 0, 720)],
            TaskOptions.SUB_AREAS_SOURCE: "fallback",
        }

    def get_task_option(self, _task_index, option, default=None):
        return self.values.get(option, default)

    def update_task_option(self, _task_index, option, value):
        self.values[option] = value


class _SignalStub:
    def __init__(self):
        self.messages = []

    def emit(self, message):
        self.messages.append(message)


def _text_polygon(xmin, xmax, ymin, ymax):
    return np.asarray(
        [
            [xmin, ymin],
            [xmax, ymin],
            [xmax, ymax],
            [xmin, ymax],
        ],
        dtype=np.float32,
    )


class _RecordingPredictor:
    def __init__(self, polygons_per_call):
        self.polygons_per_call = list(polygons_per_call)
        self.images = []

    def predict(self, image):
        self.images.append(image)
        polygons = self.polygons_per_call[len(self.images) - 1]
        return [{"dt_polys": np.asarray(polygons, dtype=np.float32)}]


class SubtitleDetectionRoiTests(unittest.TestCase):
    @staticmethod
    def _detector(sub_areas):
        detector = object.__new__(SubtitleDetect)
        detector.sub_areas = sub_areas
        return detector

    def test_selected_area_is_cropped_before_detection_and_mapped_back(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        predictor = _RecordingPredictor(
            [[_text_polygon(5, 25, 4, 14)]],
        )
        detector = self._detector([(60, 90, 20, 180)])

        with mock.patch.object(subtitle_detect, "_get_text_detector", return_value=predictor):
            boxes = detector.detect_subtitle(frame)

        self.assertEqual(predictor.images[0].shape, (30, 160, 3))
        self.assertTrue(np.shares_memory(predictor.images[0], frame))
        self.assertEqual(boxes, [(25, 45, 64, 74)])

    def test_multiple_areas_are_clipped_and_each_box_uses_its_own_offset(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        predictor = _RecordingPredictor(
            [
                [_text_polygon(1, 11, 2, 8)],
                [_text_polygon(2, 12, 3, 9)],
            ],
        )
        detector = self._detector(
            [
                (-10, 20, -5, 30),
                (70, 120, 160, 210),
                (30, 30, 5, 10),
            ],
        )

        with mock.patch.object(subtitle_detect, "_get_text_detector", return_value=predictor):
            boxes = detector.detect_subtitle(frame)

        self.assertEqual([image.shape for image in predictor.images], [(20, 30, 3), (30, 40, 3)])
        self.assertEqual(boxes, [(1, 11, 2, 8), (162, 172, 73, 79)])

    def test_no_selected_area_keeps_full_frame_detection(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        predictor = _RecordingPredictor(
            [[_text_polygon(3, 13, 5, 15)]],
        )
        detector = self._detector([])

        with mock.patch.object(subtitle_detect, "_get_text_detector", return_value=predictor):
            boxes = detector.detect_subtitle(frame)

        self.assertIs(predictor.images[0], frame)
        self.assertEqual(boxes, [(3, 13, 5, 15)])

    def test_explicit_invalid_areas_do_not_fall_back_to_full_frame(self):
        frame = np.zeros((40, 60, 3), dtype=np.uint8)
        predictor = _RecordingPredictor([])
        detector = self._detector(
            [
                (10, 10, 0, 20),
                (float("nan"), 20, 0, 20),
                ("invalid", 20, 0, 20),
            ],
        )

        with mock.patch.object(subtitle_detect, "_get_text_detector", return_value=predictor):
            boxes = detector.detect_subtitle(frame)

        self.assertEqual(predictor.images, [])
        self.assertEqual(boxes, [])


class AutoSubtitleAreaTests(unittest.TestCase):
    @staticmethod
    def _make_preprocessing_coordinator(paused=False):
        fake_home = types.SimpleNamespace(
            _auto_area_condition=threading.Condition(),
            _auto_area_preprocessing_paused=paused,
            _auto_area_active_jobs=0,
        )
        fake_home._run_auto_area_detection = types.MethodType(
            HomeInterface._run_auto_area_detection,
            fake_home,
        )
        fake_home._pause_auto_area_preprocessing = types.MethodType(
            HomeInterface._pause_auto_area_preprocessing,
            fake_home,
        )
        fake_home._resume_auto_area_preprocessing = types.MethodType(
            HomeInterface._resume_auto_area_preprocessing,
            fake_home,
        )
        return fake_home

    def test_background_detection_waits_until_preprocessing_resumes(self):
        fake_home = self._make_preprocessing_coordinator(paused=True)
        detector_started = threading.Event()

        def detect(video_path):
            detector_started.set()
            return [video_path], 1.0

        with mock.patch.object(home_interface, "auto_detect_subtitle_area", side_effect=detect):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fake_home._run_auto_area_detection, "episode-2.mp4")
                time.sleep(0.05)
                self.assertFalse(detector_started.is_set())
                self.assertFalse(future.done())

                fake_home._resume_auto_area_preprocessing()
                self.assertEqual(future.result(timeout=1), (["episode-2.mp4"], 1.0))

        self.assertTrue(detector_started.is_set())
        self.assertEqual(fake_home._auto_area_active_jobs, 0)

    def test_pausing_preprocessing_waits_for_active_detection_to_finish(self):
        fake_home = self._make_preprocessing_coordinator()
        detector_started = threading.Event()
        release_detector = threading.Event()

        def detect(_video_path):
            detector_started.set()
            release_detector.wait(timeout=1)
            return [], 0.0

        with mock.patch.object(home_interface, "auto_detect_subtitle_area", side_effect=detect):
            with ThreadPoolExecutor(max_workers=2) as executor:
                detector_future = executor.submit(fake_home._run_auto_area_detection, "episode-1.mp4")
                self.assertTrue(detector_started.wait(timeout=1))
                pause_future = executor.submit(fake_home._pause_auto_area_preprocessing)
                time.sleep(0.05)
                self.assertFalse(pause_future.done())

                release_detector.set()
                detector_future.result(timeout=1)
                pause_future.result(timeout=1)

        self.assertTrue(fake_home._auto_area_preprocessing_paused)
        self.assertEqual(fake_home._auto_area_active_jobs, 0)

    def test_next_episode_prefetch_waits_for_sttn_detection_stage(self):
        fake_home = types.SimpleNamespace(
            current_processing_task_index=3,
            _auto_area_prefetch_started_for_task=None,
            _resume_auto_area_preprocessing=mock.Mock(),
            _schedule_pending_auto_area_detections=mock.Mock(),
        )
        fake_home._maybe_start_auto_area_prefetch = types.MethodType(
            HomeInterface._maybe_start_auto_area_prefetch,
            fake_home,
        )

        self.assertFalse(
            fake_home._maybe_start_auto_area_prefetch(
                49,
                inpaint_mode=home_interface.InpaintMode.STTN_DET,
            )
        )
        self.assertTrue(
            fake_home._maybe_start_auto_area_prefetch(
                50,
                inpaint_mode=home_interface.InpaintMode.STTN_DET,
            )
        )
        self.assertFalse(
            fake_home._maybe_start_auto_area_prefetch(
                75,
                inpaint_mode=home_interface.InpaintMode.STTN_DET,
            )
        )
        fake_home._resume_auto_area_preprocessing.assert_called_once_with()
        fake_home._schedule_pending_auto_area_detections.assert_called_once_with()

    def test_sttn_auto_prefetch_can_start_on_first_real_progress(self):
        fake_home = types.SimpleNamespace(
            current_processing_task_index=1,
            _auto_area_prefetch_started_for_task=None,
            _resume_auto_area_preprocessing=mock.Mock(),
            _schedule_pending_auto_area_detections=mock.Mock(),
        )
        fake_home._maybe_start_auto_area_prefetch = types.MethodType(
            HomeInterface._maybe_start_auto_area_prefetch,
            fake_home,
        )

        self.assertTrue(
            fake_home._maybe_start_auto_area_prefetch(
                1,
                inpaint_mode=home_interface.InpaintMode.STTN_AUTO,
            )
        )
        fake_home._schedule_pending_auto_area_detections.assert_called_once_with()

    def test_prefetch_skips_current_and_already_prepared_tasks(self):
        tasks = [
            (1, types.SimpleNamespace(path="current.mp4")),
            (2, types.SimpleNamespace(path="prepared.mp4")),
            (3, types.SimpleNamespace(path="next.mp4")),
        ]
        fake_home = types.SimpleNamespace(
            current_processing_task_index=1,
            task_list_component=types.SimpleNamespace(
                get_pending_tasks=lambda: tasks,
            ),
            _task_needs_auto_area=lambda task_index, _path: task_index == 3,
            _schedule_auto_area_detection=mock.Mock(),
        )

        HomeInterface._schedule_pending_auto_area_detections(fake_home)

        fake_home._schedule_auto_area_detection.assert_called_once_with(
            3,
            "next.mp4",
        )

    def test_cached_predictor_calls_are_serialized(self):
        predictor = _NonReentrantPredictor()
        detector = object.__new__(SubtitleDetect)
        detector.sub_areas = []
        frame = np.zeros((32, 32, 3), dtype=np.uint8)

        with mock.patch.object(subtitle_detect, "_get_text_detector", return_value=predictor):
            with ThreadPoolExecutor(max_workers=2) as executor:
                futures = [executor.submit(detector.detect_subtitle, frame) for _ in range(2)]
                self.assertEqual([future.result() for future in futures], [[], []])

        self.assertEqual(predictor.max_active, 1)

    def test_default_fallback_is_a_bottom_band_not_the_full_frame(self):
        area = get_default_subtitle_area(720, 1280)

        self.assertEqual(area, (921, 1216, 36, 684))
        self.assertNotEqual(area, (0, 1280, 0, 720))
        self.assertLess(area[1] - area[0], 1280 * 0.3)

    def test_pixel_coordinates_are_not_valid_preview_coordinates(self):
        self.assertFalse(_are_normalized_preview_areas([(0, 1280, 0, 720)]))
        self.assertTrue(_are_normalized_preview_areas([(0.72, 0.95, 0.05, 0.95)]))

    def test_invalid_saved_area_falls_back_to_normalized_bottom_band(self):
        task_list = _TaskListStub()
        log_signal = _SignalStub()
        applied = {}
        fake_home = types.SimpleNamespace(
            frame_width=720,
            frame_height=1280,
            task_list_component=task_list,
            append_log_signal=log_signal,
        )
        fake_home._task_needs_auto_area = lambda _task_index, _video_path: False

        def apply_detected(task_index, areas, confidence, source="auto", log_prefix=""):
            applied.update(
                task_index=task_index,
                areas=areas,
                confidence=confidence,
                source=source,
            )
            ymin, ymax, xmin, xmax = areas[0]
            preview = [(ymin / 1280, ymax / 1280, xmin / 720, xmax / 720)]
            task_list.update_task_option(task_index, TaskOptions.SUB_AREAS, preview)
            task_list.update_task_option(task_index, TaskOptions.SUB_AREAS_SOURCE, source)
            return preview

        fake_home._apply_detected_areas_to_task = apply_detected
        fake_home._apply_default_subtitle_area_to_task = types.MethodType(
            HomeInterface._apply_default_subtitle_area_to_task,
            fake_home,
        )

        result = HomeInterface.ensure_subtitle_areas_before_run(fake_home, 0, "sample.mp4")

        self.assertTrue(_are_normalized_preview_areas(result))
        self.assertEqual(applied["areas"], [(921, 1216, 36, 684)])
        self.assertEqual(applied["source"], "fallback")
        self.assertNotEqual(result, [(0.0, 1.0, 0.0, 1.0)])


if __name__ == "__main__":
    unittest.main()
