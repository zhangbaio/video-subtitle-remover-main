from pathlib import Path
import random
import sys
import unittest

import numpy as np


WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

from backend.tools.inpaint_tools import (
    build_fixed_watermark_masks,
    create_mask,
    get_fixed_watermark_rois,
)
from backend.inpaint.propainter_inpaint import PropainterInpaint
from backend.tools.moving_mask import build_moving_watermark_mask_plan


class MovingWatermarkMaskTests(unittest.TestCase):
    def assert_alpha_has_identical_uint8_blend(self, actual, expected):
        """Account for OpenCV's multithreaded DIST_MASK_PRECISE low bits.

        Repeated calls to the existing mask builder can differ by one float32
        ULP even for the same input.  Verify the useful contract instead: the
        masks must blend identically for every possible pair of uint8 source
        and generated channel values.
        """
        if np.array_equal(actual, expected):
            return

        np.testing.assert_allclose(
            actual,
            expected,
            rtol=0.0,
            atol=np.finfo(np.float32).eps,
        )
        differing = actual != expected
        alpha_pairs = np.unique(
            np.stack((actual[differing], expected[differing]), axis=1),
            axis=0,
        )
        generated = np.arange(256, dtype=np.float32)[:, np.newaxis]
        source = np.arange(256, dtype=np.float32)[np.newaxis, :]
        for actual_alpha, expected_alpha in alpha_pairs:
            actual_blend = np.clip(
                generated * actual_alpha + source * (1.0 - actual_alpha),
                0,
                255,
            ).astype(np.uint8)
            expected_blend = np.clip(
                generated * expected_alpha + source * (1.0 - expected_alpha),
                0,
                255,
            ).astype(np.uint8)
            np.testing.assert_array_equal(actual_blend, expected_blend)

    def assert_plan_matches_full_frame_path(
        self,
        shape,
        areas,
        *,
        erase_margin=2,
        feather_radius=12,
        context=None,
        multiple=8,
    ):
        full_masks = [
            create_mask(
                shape,
                []
                if area is None
                else [(area[2], area[3], area[0], area[1])],
                deviation=0,
            )
            for area in areas
        ]
        if full_masks:
            union_selection = np.maximum.reduce(full_masks)
        else:
            union_selection = np.zeros(shape, dtype=np.uint8)
        _, union_outer, _ = build_fixed_watermark_masks(
            union_selection,
            erase_margin=erase_margin,
            feather_radius=feather_radius,
        )
        expected_rois = get_fixed_watermark_rois(
            union_outer,
            context=context,
            multiple=multiple,
        )

        actual = build_moving_watermark_mask_plan(
            shape,
            areas,
            erase_margin=erase_margin,
            feather_radius=feather_radius,
            context=context,
            multiple=multiple,
        )
        self.assertEqual(actual.rois, expected_rois)
        self.assertEqual(len(actual.outer_masks_by_roi), len(expected_rois))
        self.assertEqual(len(actual.alpha_masks_by_roi), len(expected_rois))

        for roi_index, (ymin, ymax, xmin, xmax) in enumerate(expected_rois):
            for frame_index, full_mask in enumerate(full_masks):
                _, expected_outer, expected_alpha = build_fixed_watermark_masks(
                    full_mask[ymin:ymax, xmin:xmax],
                    erase_margin=erase_margin,
                    feather_radius=feather_radius,
                    reference_shape=full_mask.shape,
                )
                np.testing.assert_array_equal(
                    actual.outer_masks_by_roi[roi_index][frame_index],
                    expected_outer,
                )
                self.assert_alpha_has_identical_uint8_blend(
                    actual.alpha_masks_by_roi[roi_index][frame_index],
                    expected_alpha,
                )

    def test_randomized_masks_are_pixel_exact(self):
        randomizer = random.Random(20260713)
        for _ in range(35):
            height = randomizer.randint(17, 193)
            width = randomizer.randint(19, 257)
            frame_count = randomizer.randint(1, 9)
            areas = []
            for _ in range(frame_count):
                if randomizer.random() < 0.12:
                    areas.append(None)
                    continue
                x1 = randomizer.randint(-8, width + 5)
                y1 = randomizer.randint(-8, height + 5)
                x2 = x1 + randomizer.randint(0, max(1, width // 4))
                y2 = y1 + randomizer.randint(0, max(1, height // 4))
                areas.append((y1, y2, x1, x2))

            with self.subTest(shape=(height, width), areas=areas):
                self.assert_plan_matches_full_frame_path(
                    (height, width),
                    areas,
                    erase_margin=randomizer.choice((0, 1, 2, 5)),
                    feather_radius=randomizer.choice((2, 5, 12, 17)),
                    context=randomizer.choice((None, 0, 3, 19)),
                    multiple=randomizer.choice((1, 8, 16)),
                )

    def test_disconnected_rois_alignment_and_inclusive_edges(self):
        shape = (109, 157)
        areas = [
            (0, 0, 0, 0),
            (2, 11, 1, 8),
            (97, 108, 145, 156),
            (101, 112, 149, 160),
        ]
        actual = build_moving_watermark_mask_plan(
            shape,
            areas,
            erase_margin=0,
            feather_radius=2,
            context=1,
            multiple=8,
        )
        self.assertEqual(len(actual.rois), 2)
        self.assert_plan_matches_full_frame_path(
            shape,
            areas,
            erase_margin=0,
            feather_radius=2,
            context=1,
            multiple=8,
        )

    def test_full_frame_scaling_matches_at_hd_resolution(self):
        self.assert_plan_matches_full_frame_path(
            (1080, 1920),
            [
                (0, 23, 0, 31),
                (500, 525, 900, 930),
                (1054, 1079, 1888, 1919),
            ],
            erase_margin=5,
            feather_radius=17,
            context=None,
            multiple=8,
        )

    def test_empty_batch_has_no_rois(self):
        self.assert_plan_matches_full_frame_path((37, 61), [])

    def test_propainter_moving_area_path_matches_full_frame_masks(self):
        rng = np.random.default_rng(20260713)
        shape = (73, 121)
        areas = [
            (11, 24, 7, 29),
            (13, 26, 10, 32),
            (15, 28, 13, 35),
        ]
        frames = [
            rng.integers(0, 256, (*shape, 3), dtype=np.uint8)
            for _ in areas
        ]
        full_masks = [
            create_mask(
                shape,
                [(area[2], area[3], area[0], area[1])],
                deviation=0,
            )
            for area in areas
        ]

        model = PropainterInpaint.__new__(PropainterInpaint)

        def deterministic_inpaint(cropped_frames, masks, **_kwargs):
            completed = []
            for frame, mask in zip(cropped_frames, masks):
                generated = np.full_like(frame, 173)
                generated[mask == 0] = frame[mask == 0]
                completed.append(generated)
            return completed

        model.inpaint = deterministic_inpaint
        expected = model.inpaint_fixed_watermark(frames, full_masks)
        actual = model.inpaint_fixed_watermark(frames, moving_areas=areas)
        for actual_frame, expected_frame in zip(actual, expected):
            np.testing.assert_array_equal(actual_frame, expected_frame)


if __name__ == "__main__":
    unittest.main()
