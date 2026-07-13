"""Memory-efficient masks for batches of tracked moving-watermark areas.

The regular fixed-watermark path accepts one full-frame selection mask per
frame.  A moving watermark only needs one rectangle per frame, so retaining
all of those full-frame masks wastes a considerable amount of memory.  This
module preserves the fixed-watermark mask semantics while materialising only
one temporary full-frame union and the masks local to each final ROI.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np

from backend.tools.inpaint_tools import (
    build_fixed_watermark_masks,
    get_fixed_watermark_rois,
)


# Tracker/public ordering: (ymin, ymax, xmin, xmax).  This deliberately differs
# from create_mask's historical x-first ordering so callers can pass the
# tracker result directly without a silent axis swap.
MovingWatermarkArea = Tuple[int, int, int, int]
FixedWatermarkRoi = Tuple[int, int, int, int]


@dataclass
class MovingWatermarkMaskPlan:
    """ROI-local masks indexed as ``[roi_index][frame_index]``."""

    rois: List[FixedWatermarkRoi]
    outer_masks_by_roi: List[List[np.ndarray]]
    alpha_masks_by_roi: List[List[np.ndarray]]


def _draw_area(
    mask: np.ndarray,
    area: Optional[MovingWatermarkArea],
    *,
    origin_x: int = 0,
    origin_y: int = 0,
    value: int = 1,
) -> None:
    """Draw exactly as ``create_mask(..., deviation=0)`` into a local view.

    In particular, OpenCV rectangles include their bottom-right coordinate.
    Translating both endpoints before drawing also lets OpenCV retain its
    clipping and reversed-endpoint behaviour at image and ROI boundaries.
    """
    if area is None:
        return
    if len(area) != 4:
        raise ValueError(f"Expected a four-value moving-watermark area, got {area!r}")

    ymin, ymax, xmin, xmax = (int(coordinate) for coordinate in area)
    x1 = max(0, xmin) - origin_x
    y1 = max(0, ymin) - origin_y
    x2 = xmax - origin_x
    y2 = ymax - origin_y
    cv2.rectangle(mask, (x1, y1), (x2, y2), value, thickness=-1)


def _build_union_outer_mask(
    frame_shape: Tuple[int, int],
    areas: Sequence[Optional[MovingWatermarkArea]],
    erase_margin: float,
    feather_radius: float,
) -> np.ndarray:
    """Build only the union layer needed for ROI discovery.

    This is the ``outer`` result of ``build_fixed_watermark_masks`` without
    computing its unused full-frame distance transform and alpha layer.
    """
    height, width = frame_shape
    raw_union = np.zeros((height, width), dtype=np.uint8)
    for area in areas:
        _draw_area(raw_union, area)

    if not np.any(raw_union):
        return raw_union

    scale = max(0.25, min(height, width) / 720.0)
    scaled_erase_margin = max(0, int(round(erase_margin * scale)))
    scaled_feather_radius = max(2, int(round(feather_radius * scale)))

    if scaled_erase_margin > 0:
        erase_kernel = cv2.getStructuringElement(
            cv2.MORPH_ELLIPSE,
            (scaled_erase_margin * 2 + 1, scaled_erase_margin * 2 + 1),
        )
        cv2.dilate(raw_union, erase_kernel, dst=raw_union)

    feather_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (scaled_feather_radius * 2 + 1, scaled_feather_radius * 2 + 1),
    )
    cv2.dilate(raw_union, feather_kernel, dst=raw_union)
    return raw_union


def build_moving_watermark_mask_plan(
    frame_shape: Sequence[int],
    areas: Sequence[Optional[MovingWatermarkArea]],
    *,
    erase_margin: float = 2,
    feather_radius: float = 12,
    context: Optional[int] = None,
    multiple: int = 8,
) -> MovingWatermarkMaskPlan:
    """Convert tracked areas directly to exact ROI-local inference masks.

    The result is equivalent to the existing dynamic fixed-watermark path:

    * each y-first tracker area is drawn on a full-frame mask with
      ``deviation=0``;
    * the union's outer layer determines connected, padded, aligned ROIs;
    * every frame's outer and alpha layers are rebuilt from its raw selection
      cropped to that ROI, with morphology scaled from the full frame.

    Only one temporary full-frame union is created.  Returned masks have ROI
    dimensions; downstream padding for frame dimensions not divisible by
    ``multiple`` remains unchanged.
    """
    if len(frame_shape) < 2:
        raise ValueError(f"Expected a frame shape with height and width, got {frame_shape!r}")
    height, width = int(frame_shape[0]), int(frame_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError(f"Frame dimensions must be positive, got {(height, width)!r}")

    normalized_areas: List[Optional[MovingWatermarkArea]] = []
    for area in areas:
        if area is None:
            normalized_areas.append(None)
            continue
        if len(area) != 4:
            raise ValueError(
                f"Expected a four-value moving-watermark area, got {area!r}"
            )
        normalized_areas.append(tuple(int(value) for value in area))

    union_outer = _build_union_outer_mask(
        (height, width),
        normalized_areas,
        erase_margin,
        feather_radius,
    )
    rois = get_fixed_watermark_rois(
        union_outer,
        context=context,
        multiple=multiple,
    )

    outer_masks_by_roi: List[List[np.ndarray]] = []
    alpha_masks_by_roi: List[List[np.ndarray]] = []
    for ymin, ymax, xmin, xmax in rois:
        local_shape = (ymax - ymin, xmax - xmin)
        roi_outer_masks: List[np.ndarray] = []
        roi_alpha_masks: List[np.ndarray] = []
        for area in normalized_areas:
            local_selection = np.zeros(local_shape, dtype=np.uint8)
            _draw_area(
                local_selection,
                area,
                origin_x=xmin,
                origin_y=ymin,
                value=255,
            )
            _, outer_mask, alpha_mask = build_fixed_watermark_masks(
                local_selection,
                erase_margin=erase_margin,
                feather_radius=feather_radius,
                reference_shape=(height, width),
            )
            roi_outer_masks.append(outer_mask)
            roi_alpha_masks.append(alpha_mask)

        outer_masks_by_roi.append(roi_outer_masks)
        alpha_masks_by_roi.append(roi_alpha_masks)

    return MovingWatermarkMaskPlan(
        rois=rois,
        outer_masks_by_roi=outer_masks_by_roi,
        alpha_masks_by_roi=alpha_masks_by_roi,
    )
