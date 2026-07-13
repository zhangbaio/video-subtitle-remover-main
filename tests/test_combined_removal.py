"""Regression tests for combined subtitle and watermark removal."""

from pathlib import Path
import sys
import unittest
from unittest import mock

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from backend import main as backend_main
from backend.main import SubtitleRemover
from backend.inpaint.propainter_inpaint import PropainterInpaint
from backend.tools.args_handler import parse_args
from backend.tools.constant import (
    InpaintMode,
    uses_fixed_watermark,
    uses_moving_watermark,
    uses_subtitles,
)
from backend.tools import subtitle_detect as subtitle_detect_module
from backend.tools.subtitle_detect import SubtitleDetect


class _FakeProgressBar:
    def __init__(self, total):
        self.total = total
        self.n = 0

    def update(self, increment):
        self.n += increment


class _FakeReader:
    def __init__(self, frames):
        self.frames = list(frames)
        self.index = 0
        self.released = False

    def read(self):
        if self.index >= len(self.frames):
            return False, None
        frame = self.frames[self.index]
        self.index += 1
        return True, frame.copy()

    def release(self):
        self.released = True


class _FakeWriter:
    def __init__(self):
        self.frames = []

    def write(self, frame):
        self.frames.append(frame.copy())


class CombinedRemovalTests(unittest.TestCase):
    @staticmethod
    def make_remover(frame_count=3, shape=(40, 60)):
        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.video_path = "video.mp4"
        remover.sub_areas = [(20, 39, 0, 59)]
        remover.watermark_areas = [(2, 8, 45, 55)]
        remover.mask_size = shape
        remover.frame_height, remover.frame_width = shape
        remover.frame_count = frame_count
        remover.fps = 25.0
        remover.ab_sections = None
        remover.video_cap = object()
        remover.video_writer = _FakeWriter()
        remover.propainter_max_load_num = 8
        remover.moving_watermark_fast_mode = True
        remover.append_output = mock.Mock()
        remover.report_processing_phase = mock.Mock()
        remover.push_preview_with_comp = mock.Mock()
        remover.update_progress = lambda tbar, increment: tbar.update(increment)
        return remover

    def test_moving_failure_keeps_subtitle_and_missing_subtitle_keeps_watermark(self):
        frames = [
            np.full((40, 60, 3), value, dtype=np.uint8)
            for value in (10, 20, 30)
        ]
        reader = _FakeReader(frames)
        remover = self.make_remover()
        captured_masks = []

        def inpaint(batch_frames, masks, **_kwargs):
            captured_masks.extend(mask.copy() for mask in masks)
            return [frame + 1 for frame in batch_frames]

        remover._inpaint_fixed_watermark_batch = inpaint
        remover._load_moving_watermark_plan = mock.Mock(
            return_value=([None, (2, 8, 45, 55), None], set(), {})
        )

        class FakeSubtitleDetect:
            def __init__(self, *_args):
                pass

            @staticmethod
            def find_subtitle_frame_no(**_kwargs):
                # SubtitleDetect's public dictionary is 1-based.
                return {1: [(5, 20, 28, 35)]}

        with (
            mock.patch.object(backend_main, "SubtitleDetect", FakeSubtitleDetect),
            mock.patch.object(backend_main, "create_processing_capture", return_value=reader) as create_capture,
            mock.patch.object(backend_main, "FramePrefetcher", side_effect=lambda capture: capture),
        ):
            progress = _FakeProgressBar(3)
            remover.combined_subtitle_watermark_mode(progress, moving=True)

        create_capture.assert_called_once()
        self.assertTrue(reader.released)
        self.assertEqual(progress.n, 3)
        self.assertEqual(len(captured_masks), 2)
        # Frame 0: tracking failed, but its 1-based OCR result still applies.
        self.assertGreater(captured_masks[0][31, 10], 0)
        self.assertEqual(captured_masks[0][4, 50], 0)
        # Frame 1: no subtitle, but the tracked watermark still applies.
        self.assertEqual(captured_masks[1][31, 10], 0)
        self.assertGreater(captured_masks[1][4, 50], 0)
        # Frame 2 has neither layer and therefore bypasses inference unchanged.
        np.testing.assert_array_equal(remover.video_writer.frames[2], frames[2])

    def test_fixed_watermark_continues_when_ocr_returns_no_subtitles(self):
        frames = [np.zeros((40, 60, 3), dtype=np.uint8) for _ in range(2)]
        reader = _FakeReader(frames)
        remover = self.make_remover(frame_count=2)
        captured_masks = []

        def inpaint(batch_frames, masks, **_kwargs):
            captured_masks.extend(mask.copy() for mask in masks)
            return list(batch_frames)

        remover._inpaint_fixed_watermark_batch = inpaint

        class FakeSubtitleDetect:
            def __init__(self, *_args):
                pass

            @staticmethod
            def find_subtitle_frame_no(**_kwargs):
                return {}

            @staticmethod
            def get_scene_div_frame_no(_path):
                return []

        with (
            mock.patch.object(backend_main, "SubtitleDetect", FakeSubtitleDetect),
            mock.patch.object(backend_main, "create_processing_capture", return_value=reader),
            mock.patch.object(backend_main, "FramePrefetcher", side_effect=lambda capture: capture),
        ):
            progress = _FakeProgressBar(2)
            remover.combined_subtitle_watermark_mode(progress, moving=False)

        self.assertEqual(len(captured_masks), 2)
        self.assertTrue(all(mask[4, 50] > 0 for mask in captured_masks))
        self.assertEqual(progress.n, 2)

    def test_ab_sections_gate_both_layers_with_zero_based_indexes(self):
        frames = [np.zeros((40, 60, 3), dtype=np.uint8) for _ in range(3)]
        reader = _FakeReader(frames)
        remover = self.make_remover()
        remover.ab_sections = [range(1, 2)]
        captured_masks = []

        def inpaint(batch_frames, masks, **_kwargs):
            captured_masks.extend(mask.copy() for mask in masks)
            return list(batch_frames)

        remover._inpaint_fixed_watermark_batch = inpaint

        class FakeSubtitleDetect:
            def __init__(self, *_args):
                pass

            @staticmethod
            def find_subtitle_frame_no(**_kwargs):
                return {
                    1: [(5, 20, 28, 35)],
                    2: [(5, 20, 28, 35)],
                    3: [(5, 20, 28, 35)],
                }

            @staticmethod
            def get_scene_div_frame_no(_path):
                return []

        with (
            mock.patch.object(backend_main, "SubtitleDetect", FakeSubtitleDetect),
            mock.patch.object(backend_main, "create_processing_capture", return_value=reader),
            mock.patch.object(backend_main, "FramePrefetcher", side_effect=lambda capture: capture),
        ):
            remover.combined_subtitle_watermark_mode(_FakeProgressBar(3), moving=False)

        self.assertEqual(len(captured_masks), 1)
        self.assertGreater(captured_masks[0][31, 10], 0)
        self.assertGreater(captured_masks[0][4, 50], 0)
        np.testing.assert_array_equal(remover.video_writer.frames[0], frames[0])
        np.testing.assert_array_equal(remover.video_writer.frames[2], frames[2])

    def test_dynamic_union_uses_independent_propainter_rois(self):
        remover = self.make_remover(frame_count=2, shape=(100, 240))
        masks = [
            remover._create_combined_mask(
                [(5, 25, 82, 92)],
                [(3, 12, 210, 224)],
            ),
            remover._create_combined_mask(
                [(7, 27, 82, 92)],
                [(4, 13, 209, 223)],
            ),
        ]
        frames = [np.zeros((100, 240, 3), dtype=np.uint8) for _ in masks]
        model = PropainterInpaint.__new__(PropainterInpaint)
        roi_calls = []

        def deterministic_inpaint(cropped_frames, cropped_masks, **_kwargs):
            roi_calls.append(cropped_frames[0].shape[:2])
            completed = []
            for frame, mask in zip(cropped_frames, cropped_masks):
                generated = np.full_like(frame, 200)
                generated[mask == 0] = frame[mask == 0]
                completed.append(generated)
            return completed

        model.inpaint = deterministic_inpaint
        completed = model.inpaint_fixed_watermark(frames, masks)

        # Top-right watermark and bottom-left subtitle remain disconnected and
        # are inferred as compact independent ROIs, not one large rectangle.
        self.assertEqual(len(roi_calls), 2)
        self.assertGreater(completed[0][7, 217, 0], 0)
        self.assertGreater(completed[0][87, 15, 0], 0)
        self.assertEqual(completed[0][50, 120, 0], 0)

    def test_combined_mode_capabilities_and_cli_regions(self):
        self.assertTrue(uses_subtitles(InpaintMode.SUBTITLE_FIXED_WATERMARK))
        self.assertTrue(uses_fixed_watermark(InpaintMode.SUBTITLE_FIXED_WATERMARK))
        self.assertTrue(uses_subtitles(InpaintMode.SUBTITLE_MOVING_WATERMARK))
        self.assertTrue(uses_moving_watermark(InpaintMode.SUBTITLE_MOVING_WATERMARK))

        argv = [
            "backend.main",
            "-i",
            "input.mp4",
            "--inpaint-mode",
            "subtitle-moving-watermark",
            "-c",
            "20",
            "40",
            "0",
            "100",
            "-w",
            "2",
            "12",
            "80",
            "96",
            "--watermark-reference-frame",
            "7",
        ]
        with mock.patch.object(sys, "argv", argv):
            args = parse_args()
        self.assertIs(args.inpaint_mode, InpaintMode.SUBTITLE_MOVING_WATERMARK)
        self.assertEqual(args.subtitle_area_coords, [[20, 40, 0, 100]])
        self.assertEqual(args.watermark_area_coords, [[2, 12, 80, 96]])
        self.assertEqual(args.watermark_reference_frame, 7)

        remover = SubtitleRemover.__new__(SubtitleRemover)
        remover.sub_areas = []
        remover.watermark_areas = [tuple(args.watermark_area_coords[0])]
        self.assertEqual(remover._get_watermark_areas(), remover.watermark_areas)

    def test_subtitle_scan_releases_capture_when_ocr_fails(self):
        class FailingCapture:
            def __init__(self):
                self.read_count = 0
                self.released = False

            @staticmethod
            def get(_property):
                return 1

            def isOpened(self):
                return not self.released

            def read(self):
                if self.read_count:
                    return False, None
                self.read_count += 1
                return True, np.zeros((8, 8, 3), dtype=np.uint8)

            def release(self):
                self.released = True

        capture = FailingCapture()
        detector = SubtitleDetect.__new__(SubtitleDetect)
        detector.video_path = "input.mp4"
        detector.detect_subtitle = mock.Mock(side_effect=RuntimeError("OCR failed"))

        with mock.patch.object(
            subtitle_detect_module.cv2,
            "VideoCapture",
            return_value=capture,
        ):
            with self.assertRaisesRegex(RuntimeError, "OCR failed"):
                detector.find_subtitle_frame_no()
        self.assertTrue(capture.released)


if __name__ == "__main__":
    unittest.main()
