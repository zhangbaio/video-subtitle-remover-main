from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class LetterboxTransform:
    resized_width: int
    resized_height: int
    left: int
    top: int


def create_fixed_watermark_mask(
    frame_shape: Sequence[int],
    areas: Iterable[Sequence[int]],
) -> np.ndarray:
    """Create an exact full-frame mask from ``(ymin, ymax, xmin, xmax)`` areas."""
    height, width = int(frame_shape[0]), int(frame_shape[1])
    mask = np.zeros((height, width), dtype=np.uint8)
    for area in areas:
        if len(area) != 4:
            continue
        ymin, ymax, xmin, xmax = (int(round(value)) for value in area)
        ymin, ymax = sorted((max(0, ymin), min(height, ymax)))
        xmin, xmax = sorted((max(0, xmin), min(width, xmax)))
        if ymin < ymax and xmin < xmax:
            mask[ymin:ymax, xmin:xmax] = 255
    return mask


class FixedWatermarkInpaint:
    """Apply STTN only to small, fixed watermark regions.

    The wrapped STTN detector model already accepts an explicit mask.  This
    adapter avoids its full-width subtitle crop: each connected watermark mask
    is cropped with local context, letterboxed without changing aspect ratio,
    processed temporally, then composited back only through a feathered mask.
    """

    def __init__(
        self,
        sttn_inpaint,
        *,
        mask_expansion: int = 2,
        feather_radius: float = 1.5,
    ):
        self.sttn_inpaint = sttn_inpaint
        self.model_width = int(sttn_inpaint.model_input_width)
        self.model_height = int(sttn_inpaint.model_input_height)
        self.mask_expansion = max(0, int(mask_expansion))
        self.feather_radius = max(0.0, float(feather_radius))

    @staticmethod
    def _component_masks(mask: np.ndarray) -> List[np.ndarray]:
        binary = (mask > 0).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, 8)
        components = []
        for label in range(1, count):
            if stats[label, cv2.CC_STAT_AREA] <= 0:
                continue
            component = np.zeros_like(binary)
            component[labels == label] = 1
            components.append(component)
        return components

    @staticmethod
    def _context_box(mask: np.ndarray) -> Tuple[int, int, int, int]:
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0, 0, 0, 0

        height, width = mask.shape[:2]
        xmin, xmax = int(xs.min()), int(xs.max()) + 1
        ymin, ymax = int(ys.min()), int(ys.max()) + 1
        region_width = xmax - xmin
        region_height = ymax - ymin
        pad_x = max(24, int(round(region_width * 0.15)))
        pad_y = max(16, int(round(region_height * 0.75)))
        return (
            max(0, ymin - pad_y),
            min(height, ymax + pad_y),
            max(0, xmin - pad_x),
            min(width, xmax + pad_x),
        )

    @staticmethod
    def _letterbox(
        image: np.ndarray,
        target_width: int,
        target_height: int,
        *,
        is_mask: bool,
    ) -> Tuple[np.ndarray, LetterboxTransform]:
        source_height, source_width = image.shape[:2]
        scale = min(target_width / source_width, target_height / source_height)
        resized_width = max(1, min(target_width, int(round(source_width * scale))))
        resized_height = max(1, min(target_height, int(round(source_height * scale))))
        interpolation = cv2.INTER_NEAREST if is_mask else cv2.INTER_LINEAR
        resized = cv2.resize(
            image,
            (resized_width, resized_height),
            interpolation=interpolation,
        )
        left = (target_width - resized_width) // 2
        right = target_width - resized_width - left
        top = (target_height - resized_height) // 2
        bottom = target_height - resized_height - top
        if is_mask:
            padded = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_CONSTANT,
                value=0,
            )
        else:
            padded = cv2.copyMakeBorder(
                resized,
                top,
                bottom,
                left,
                right,
                cv2.BORDER_REFLECT_101,
            )
        return padded, LetterboxTransform(
            resized_width=resized_width,
            resized_height=resized_height,
            left=left,
            top=top,
        )

    @staticmethod
    def _restore_letterbox(
        image: np.ndarray,
        transform: LetterboxTransform,
        output_width: int,
        output_height: int,
    ) -> np.ndarray:
        unpadded = image[
            transform.top:transform.top + transform.resized_height,
            transform.left:transform.left + transform.resized_width,
        ]
        return cv2.resize(
            unpadded,
            (output_width, output_height),
            interpolation=cv2.INTER_LINEAR,
        )

    def __call__(
        self,
        input_frames: List[np.ndarray],
        input_mask: np.ndarray,
        active_frames: Sequence[bool] | None = None,
    ) -> List[np.ndarray]:
        if not input_frames:
            return []
        if active_frames is None:
            active_frames = [True] * len(input_frames)
        if len(active_frames) != len(input_frames):
            raise ValueError("active_frames must match input_frames")

        results = [frame.copy() for frame in input_frames]
        if not any(active_frames):
            return results

        binary_mask = (np.asarray(input_mask) > 0).astype(np.uint8)
        if not np.any(binary_mask):
            return results

        for component in self._component_masks(binary_mask):
            if self.mask_expansion:
                size = self.mask_expansion * 2 + 1
                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
                model_mask = cv2.dilate(component, kernel)
            else:
                model_mask = component

            ymin, ymax, xmin, xmax = self._context_box(model_mask)
            if ymin >= ymax or xmin >= xmax:
                continue

            crop_height, crop_width = ymax - ymin, xmax - xmin
            prepared_frames = []
            frame_transform = None
            for frame in results:
                prepared, frame_transform = self._letterbox(
                    frame[ymin:ymax, xmin:xmax],
                    self.model_width,
                    self.model_height,
                    is_mask=False,
                )
                prepared_frames.append(prepared)

            prepared_mask, mask_transform = self._letterbox(
                model_mask[ymin:ymax, xmin:xmax],
                self.model_width,
                self.model_height,
                is_mask=True,
            )
            # STTN's torchvision transform normalizes uint8 masks to [0, 1]
            # before thresholding, so the prepared mask must remain 0/255.
            prepared_mask = (prepared_mask > 0).astype(np.uint8) * 255
            if all(active_frames):
                inference_masks = prepared_mask
            else:
                empty_mask = np.zeros_like(prepared_mask)
                inference_masks = [
                    prepared_mask if active else empty_mask
                    for active in active_frames
                ]

            completed_frames = self.sttn_inpaint.inpaint(
                prepared_frames,
                inference_masks,
            )

            alpha = model_mask[ymin:ymax, xmin:xmax].astype(np.float32)
            if self.feather_radius:
                alpha = cv2.GaussianBlur(
                    alpha,
                    (0, 0),
                    sigmaX=self.feather_radius,
                    sigmaY=self.feather_radius,
                )
            alpha = np.clip(alpha, 0.0, 1.0)[:, :, None]

            for index, active in enumerate(active_frames):
                if not active:
                    continue
                restored = self._restore_letterbox(
                    completed_frames[index],
                    frame_transform or mask_transform,
                    crop_width,
                    crop_height,
                ).astype(np.float32)
                original_crop = results[index][ymin:ymax, xmin:xmax].astype(np.float32)
                blended = restored * alpha + original_crop * (1.0 - alpha)
                results[index][ymin:ymax, xmin:xmax] = np.clip(
                    blended,
                    0,
                    255,
                ).astype(np.uint8)

        return results
