import threading
import time
import types
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest import mock

import numpy as np

from backend.tools import subtitle_detect
from backend.tools.subtitle_detect import (
    SubtitleDetect,
    SubtitleDetectionCancelled,
    auto_detect_subtitle_area,
    get_default_subtitle_area,
)
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


class _BatchRecordingPredictor:
    def __init__(self, polygons_per_image):
        self.polygons_per_image = list(polygons_per_image)
        self.calls = []

    def predict(self, images, batch_size=None):
        images = list(images)
        self.calls.append((images, batch_size))
        if len(images) > len(self.polygons_per_image):
            raise AssertionError("unexpected batch length")
        polygons_for_call = self.polygons_per_image[:len(images)]
        del self.polygons_per_image[:len(images)]
        return [
            {"dt_polys": np.asarray(polygons, dtype=np.float32)}
            for polygons in polygons_for_call
        ]


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

    def test_roi_batch_preserves_frame_order_offsets_and_per_frame_deduplication(self):
        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8),
        ]
        predictor = _BatchRecordingPredictor(
            [
                [_text_polygon(15, 25, 2, 8)],
                [_text_polygon(20, 30, 3, 9)],
                [_text_polygon(5, 15, 2, 8)],
                [],
            ]
        )
        detector = self._detector(
            [
                (60, 90, 10, 70),
                (60, 90, 20, 80),
            ]
        )

        with mock.patch.object(subtitle_detect, "_get_text_detector", return_value=predictor):
            boxes = detector.detect_subtitle_batch(frames)

        self.assertEqual(
            boxes,
            [
                [(25, 35, 62, 68)],
                [(30, 40, 63, 69)],
            ],
        )
        self.assertEqual(len(predictor.calls), 2)
        for images, batch_size in predictor.calls:
            self.assertEqual(batch_size, 4)
            self.assertEqual([image.shape for image in images], [(30, 60, 3)] * 2)
            self.assertEqual(int(images[0][0, 0, 0]), 0)
            self.assertEqual(int(images[1][0, 0, 0]), 1)

    def test_large_roi_limits_predictor_batch_size_to_two(self):
        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.zeros((100, 100, 3), dtype=np.uint8),
        ]
        predictor = _BatchRecordingPredictor([[], []])
        detector = self._detector([(0, 80, 0, 80)])

        with mock.patch.object(subtitle_detect, "_get_text_detector", return_value=predictor):
            self.assertEqual(
                detector.detect_subtitle_batch(frames, batch_size=8),
                [[], []],
            )

        self.assertEqual(predictor.calls[0][1], 2)

    def test_differently_shaped_rois_are_sent_as_separate_predictor_batches(self):
        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8),
        ]
        predictor = _BatchRecordingPredictor([[], [], [], []])
        detector = self._detector(
            [
                (60, 90, 10, 70),
                (20, 60, 5, 35),
            ]
        )

        with mock.patch.object(subtitle_detect, "_get_text_detector", return_value=predictor):
            self.assertEqual(detector.detect_subtitle_batch(frames), [[], []])

        self.assertEqual(len(predictor.calls), 2)
        self.assertEqual(
            [[image.shape for image in images] for images, _batch_size in predictor.calls],
            [[(30, 60, 3), (30, 60, 3)], [(40, 30, 3), (40, 30, 3)]],
        )
        self.assertEqual([batch_size for _images, batch_size in predictor.calls], [4, 4])

    def test_roi_batch_rejects_missing_predictor_results(self):
        frames = [
            np.zeros((100, 100, 3), dtype=np.uint8),
            np.ones((100, 100, 3), dtype=np.uint8),
        ]
        predictor = mock.Mock()
        predictor.predict.return_value = [{"dt_polys": np.empty((0, 4, 2))}]
        detector = self._detector([(60, 90, 10, 70)])

        with (
            mock.patch.object(subtitle_detect, "_get_text_detector", return_value=predictor),
            self.assertRaisesRegex(RuntimeError, "unexpected number"),
        ):
            detector.detect_subtitle_batch(frames)

    def test_full_frame_batch_helper_keeps_single_frame_predict_calls(self):
        frames = [
            np.zeros((40, 60, 3), dtype=np.uint8),
            np.ones((40, 60, 3), dtype=np.uint8),
        ]
        detector = self._detector([])
        detector.detect_subtitle = mock.Mock(side_effect=[[(1, 2, 3, 4)], []])

        self.assertEqual(
            detector.detect_subtitle_batch(frames),
            [[(1, 2, 3, 4)], []],
        )
        self.assertEqual(detector.detect_subtitle.call_count, 2)


class AutoSubtitleAreaTests(unittest.TestCase):
    @staticmethod
    def _make_preprocessing_coordinator(paused=False):
        fake_home = types.SimpleNamespace(
            _auto_area_condition=threading.Condition(),
            _auto_area_lock=threading.Lock(),
            _auto_area_preprocessing_paused=paused,
            _auto_area_active_jobs=0,
            _auto_area_cancel_event=threading.Event(),
            _auto_area_futures={},
            _auto_area_errors={},
            _auto_area_results={},
            _manual_auto_area_future=None,
            _stop_event=threading.Event(),
            _closing=False,
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

        def detect(video_path, **_kwargs):
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

        def detect(_video_path, **_kwargs):
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

    def test_paused_background_detection_is_cancelled_without_running(self):
        fake_home = self._make_preprocessing_coordinator(paused=True)
        detector = mock.Mock()

        with mock.patch.object(home_interface, "auto_detect_subtitle_area", detector):
            with ThreadPoolExecutor(max_workers=1) as executor:
                future = executor.submit(fake_home._run_auto_area_detection, "episode.mp4")
                time.sleep(0.05)
                fake_home._auto_area_cancel_event.set()
                with fake_home._auto_area_condition:
                    fake_home._auto_area_condition.notify_all()
                with self.assertRaises(SubtitleDetectionCancelled):
                    future.result(timeout=1)

        detector.assert_not_called()
        self.assertEqual(fake_home._auto_area_active_jobs, 0)

    def test_wait_for_auto_area_future_polls_stop_event(self):
        fake_home = self._make_preprocessing_coordinator()
        fake_home._stop_event.set()
        future = mock.Mock()

        with self.assertRaises(SubtitleDetectionCancelled):
            HomeInterface._wait_for_auto_area_future(fake_home, future)

        future.result.assert_not_called()

    def test_auto_area_cancellation_cancels_futures_outside_lock(self):
        fake_home = self._make_preprocessing_coordinator()
        future = mock.Mock()
        manual_future = mock.Mock()
        fake_home._auto_area_futures["episode.mp4"] = future
        fake_home._manual_auto_area_future = (
            1,
            0,
            "manual.mp4",
            fake_home._auto_area_cancel_event,
            manual_future,
        )

        cancelled = HomeInterface._cancel_auto_area_detections(fake_home, clear=True)

        self.assertTrue(fake_home._auto_area_cancel_event.is_set())
        future.cancel.assert_called_once_with()
        manual_future.cancel.assert_called_once_with()
        self.assertEqual(cancelled, [future, manual_future])
        self.assertEqual(fake_home._auto_area_futures, {})

    def test_cancelled_auto_area_wait_does_not_apply_fallback(self):
        future = mock.Mock()
        fallback = mock.Mock()
        fake_home = types.SimpleNamespace(
            task_list_component=types.SimpleNamespace(
                get_task_option=lambda *_args: [],
                update_task_option=mock.Mock(),
            ),
            _task_needs_auto_area=lambda *_args: True,
            _get_auto_area_detection_result=lambda _path: (None, future, None),
            _schedule_auto_area_detection=mock.Mock(),
            _wait_for_auto_area_future=mock.Mock(
                side_effect=SubtitleDetectionCancelled("cancelled")
            ),
            _apply_detected_areas_to_task=mock.Mock(),
            _apply_default_subtitle_area_to_task=fallback,
            append_log_signal=types.SimpleNamespace(emit=mock.Mock()),
        )

        with self.assertRaises(SubtitleDetectionCancelled):
            HomeInterface.ensure_subtitle_areas_before_run(
                fake_home,
                0,
                "episode.mp4",
            )

        fallback.assert_not_called()
        fake_home._apply_detected_areas_to_task.assert_not_called()

    def test_stop_during_future_completion_cannot_resurrect_result(self):
        future = mock.Mock()
        stop_event = threading.Event()
        results = {}
        futures = {"episode.mp4": future}

        def finish_after_stop(_future):
            stop_event.set()
            futures.clear()
            return ([(72, 95, 5, 95)], 1.0)

        fake_home = types.SimpleNamespace(
            task_list_component=types.SimpleNamespace(
                get_task_option=lambda *_args: [],
                update_task_option=mock.Mock(),
            ),
            _task_needs_auto_area=lambda *_args: True,
            _get_auto_area_detection_result=lambda _path: (None, future, None),
            _schedule_auto_area_detection=mock.Mock(),
            _wait_for_auto_area_future=finish_after_stop,
            _auto_area_lock=threading.Lock(),
            _auto_area_results=results,
            _auto_area_futures=futures,
            _auto_area_errors={},
            _auto_area_cancel_event=threading.Event(),
            _stop_event=stop_event,
            _closing=False,
            _apply_detected_areas_to_task=mock.Mock(),
            _apply_default_subtitle_area_to_task=mock.Mock(),
            append_log_signal=types.SimpleNamespace(emit=mock.Mock()),
        )

        with self.assertRaises(SubtitleDetectionCancelled):
            HomeInterface.ensure_subtitle_areas_before_run(
                fake_home,
                0,
                "episode.mp4",
            )

        self.assertEqual(results, {})
        fake_home._apply_detected_areas_to_task.assert_not_called()

    def test_manual_auto_area_result_is_discarded_after_task_switch(self):
        display = types.SimpleNamespace(
            video_coordinates_to_preview_coordinates=mock.Mock(),
            set_selection_rects=mock.Mock(),
            save_selections_to_config=mock.Mock(),
        )
        fake_home = types.SimpleNamespace(
            _auto_area_lock=threading.Lock(),
            _closing=False,
            _manual_auto_area_generation=7,
            video_path="episode-b.mp4",
            task_list_component=types.SimpleNamespace(
                get_current_task_index=lambda: 2,
                get_task=mock.Mock(),
                update_task_option=mock.Mock(),
            ),
            video_display_component=display,
            append_output=mock.Mock(),
        )
        fake_home._manual_auto_area_target_is_current = types.MethodType(
            HomeInterface._manual_auto_area_target_is_current,
            fake_home,
        )

        HomeInterface.on_auto_subtitle_area_detected(
            fake_home,
            1,
            "episode-a.mp4",
            7,
            [(72, 95, 5, 95)],
            1.0,
        )

        display.video_coordinates_to_preview_coordinates.assert_not_called()
        fake_home.task_list_component.update_task_option.assert_not_called()

    def test_unopened_preview_capture_is_explicitly_released(self):
        capture = mock.Mock()
        capture.isOpened.return_value = False
        fake_home = types.SimpleNamespace(
            video_path=None,
            video_cap=None,
            _video_cap_lock=threading.Lock(),
            load_as_picture=mock.Mock(return_value=False),
        )

        with (
            mock.patch.object(home_interface, "is_image_file", return_value=False),
            mock.patch.object(home_interface, "get_readable_path", side_effect=lambda path: path),
            mock.patch.object(home_interface.cv2, "VideoCapture", return_value=capture),
        ):
            self.assertFalse(HomeInterface.load_video(fake_home, "episode.mp4"))

        capture.release.assert_called_once_with()
        self.assertIsNone(fake_home.video_cap)

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

    def test_auto_area_cancel_releases_video_capture(self):
        cancel_event = threading.Event()
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        capture = mock.Mock()
        capture.isOpened.return_value = True

        def get_property(prop):
            values = {
                subtitle_detect.cv2.CAP_PROP_FRAME_COUNT: 3,
                subtitle_detect.cv2.CAP_PROP_FRAME_WIDTH: 200,
                subtitle_detect.cv2.CAP_PROP_FRAME_HEIGHT: 100,
            }
            return values.get(prop, 0)

        capture.get.side_effect = get_property
        capture.read.return_value = (True, frame)
        detector = mock.Mock()

        def detect(_frame):
            cancel_event.set()
            return []

        detector.detect_subtitle.side_effect = detect
        with (
            mock.patch.object(subtitle_detect.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(subtitle_detect, "SubtitleDetect", return_value=detector),
            self.assertRaises(SubtitleDetectionCancelled),
        ):
            auto_detect_subtitle_area(
                "episode.mp4",
                sample_count=3,
                cancel_event=cancel_event,
            )

        capture.release.assert_called_once_with()
        detector.detect_subtitle.assert_called_once()

    def test_auto_area_detector_error_releases_video_capture(self):
        frame = np.zeros((100, 200, 3), dtype=np.uint8)
        capture = mock.Mock()
        capture.isOpened.return_value = True
        capture.get.side_effect = lambda prop: {
            subtitle_detect.cv2.CAP_PROP_FRAME_COUNT: 1,
            subtitle_detect.cv2.CAP_PROP_FRAME_WIDTH: 200,
            subtitle_detect.cv2.CAP_PROP_FRAME_HEIGHT: 100,
        }.get(prop, 0)
        capture.read.return_value = (True, frame)
        detector = mock.Mock()
        detector.detect_subtitle.side_effect = RuntimeError("OCR failed")

        with (
            mock.patch.object(subtitle_detect.cv2, "VideoCapture", return_value=capture),
            mock.patch.object(subtitle_detect, "SubtitleDetect", return_value=detector),
            self.assertRaisesRegex(RuntimeError, "OCR failed"),
        ):
            auto_detect_subtitle_area("episode.mp4", sample_count=1)

        capture.release.assert_called_once_with()

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
