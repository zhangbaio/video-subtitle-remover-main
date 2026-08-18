import unittest

import cv2
import numpy as np

from backend.inpaint.fixed_watermark_inpaint import (
    FixedWatermarkInpaint,
    create_fixed_watermark_mask,
)


class _MaskedFillModel:
    model_input_width = 432
    model_input_height = 240

    def __init__(self):
        self.calls = []

    def inpaint(self, frames, masks):
        self.calls.append((frames, masks))
        if isinstance(masks, np.ndarray):
            masks = [masks] * len(frames)
        output = []
        for frame, mask in zip(frames, masks):
            completed = frame.copy()
            completed[np.asarray(mask) > 0] = [210, 40, 15]
            output.append(completed)
        return output


class FixedWatermarkMaskTests(unittest.TestCase):
    def test_mask_clamps_and_ignores_invalid_areas(self):
        mask = create_fixed_watermark_mask(
            (8, 10),
            [(-2, 3, 7, 20), (5, 5, 1, 4), (1, 2, 3)],
        )

        self.assertEqual(mask.shape, (8, 10))
        self.assertEqual(int(mask.sum() / 255), 9)
        self.assertTrue(np.all(mask[0:3, 7:10] == 255))


class FixedWatermarkInpaintTests(unittest.TestCase):
    def setUp(self):
        self.model = _MaskedFillModel()
        self.inpaint = FixedWatermarkInpaint(
            self.model,
            mask_expansion=0,
            feather_radius=0,
        )

    def test_context_box_is_local_and_padded(self):
        mask = np.zeros((720, 1280), dtype=np.uint8)
        mask[24:65, 994:1256] = 1

        ymin, ymax, xmin, xmax = self.inpaint._context_box(mask)

        self.assertEqual((ymin, xmin, xmax), (0, 955, 1280))
        self.assertLessEqual(ymax, 100)
        self.assertLess(xmax - xmin, 1280)

    def test_letterbox_preserves_aspect_ratio(self):
        image = np.zeros((100, 300, 3), dtype=np.uint8)

        result, transform = self.inpaint._letterbox(
            image,
            432,
            240,
            is_mask=False,
        )

        self.assertEqual(result.shape, (240, 432, 3))
        self.assertEqual((transform.resized_width, transform.resized_height), (432, 144))
        self.assertEqual(transform.top, 48)

    def test_composite_changes_only_selected_pixels(self):
        frame = np.empty((120, 200, 3), dtype=np.uint8)
        frame[:, :] = [10, 20, 200]
        original = frame.copy()
        mask = create_fixed_watermark_mask((120, 200), [(20, 45, 150, 190)])

        result = self.inpaint([frame], mask)[0]

        np.testing.assert_array_equal(result[mask == 0], original[mask == 0])
        np.testing.assert_array_equal(
            result[mask > 0],
            np.broadcast_to([210, 40, 15], result[mask > 0].shape),
        )
        np.testing.assert_array_equal(frame, original)
        prepared_frames, _ = self.model.calls[0]
        self.assertEqual(prepared_frames[0].shape, (240, 432, 3))

    def test_inactive_frames_are_context_only(self):
        frames = [
            np.full((80, 120, 3), value, dtype=np.uint8)
            for value in (10, 20, 30)
        ]
        originals = [frame.copy() for frame in frames]
        mask = create_fixed_watermark_mask((80, 120), [(10, 30, 80, 110)])

        result = self.inpaint(frames, mask, active_frames=[True, False, True])

        np.testing.assert_array_equal(result[1], originals[1])
        self.assertFalse(np.array_equal(result[0], originals[0]))
        self.assertFalse(np.array_equal(result[2], originals[2]))
        _, masks = self.model.calls[0]
        self.assertEqual(len(masks), 3)
        self.assertFalse(np.any(masks[1]))

    def test_disconnected_regions_are_processed_separately(self):
        frame = np.zeros((100, 180, 3), dtype=np.uint8)
        mask = create_fixed_watermark_mask(
            (100, 180),
            [(10, 20, 10, 30), (70, 85, 140, 170)],
        )

        self.inpaint([frame], mask)

        self.assertEqual(len(self.model.calls), 2)


if __name__ == "__main__":
    unittest.main()
