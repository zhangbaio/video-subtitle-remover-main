import subprocess
from collections import deque

import cv2
import numpy as np

from backend.tools.common_tools import get_readable_path
from backend.tools.ffmpeg_cli import FFmpegCLI


def clamp_area(area, frame_shape):
    height, width = frame_shape[:2]
    ymin, ymax, xmin, xmax = map(int, area)
    ymin = max(0, min(height, ymin))
    ymax = max(0, min(height, ymax))
    xmin = max(0, min(width, xmin))
    xmax = max(0, min(width, xmax))
    if ymax <= ymin or xmax <= xmin:
        return None
    return ymin, ymax, xmin, xmax


def read_video_frame(video_path, frame_no=0, fps=0.0):
    """Read an exact reference frame, with an FFmpeg fallback for Unicode paths."""
    frame_no = max(0, int(frame_no or 0))
    readable_path = get_readable_path(video_path) or video_path
    cap = cv2.VideoCapture(readable_path)
    source_fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
    if cap.isOpened():
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
        ok, frame = cap.read()
        cap.release()
        if ok and frame is not None:
            return frame
    else:
        cap.release()

    fps = source_fps or float(fps or 0.0)
    seek_seconds = frame_no / fps if fps > 0 else 0
    command = [
        FFmpegCLI.instance().ffmpeg_path,
        "-hide_banner",
        "-loglevel", "error",
        "-ss", f"{seek_seconds:.6f}",
        "-i", video_path,
        "-frames:v", "1",
        "-f", "image2pipe",
        "-vcodec", "png",
        "-",
    ]
    try:
        encoded = subprocess.check_output(command, stdin=subprocess.DEVNULL, timeout=60)
    except Exception:
        return None
    return cv2.imdecode(np.frombuffer(encoded, dtype=np.uint8), cv2.IMREAD_COLOR)


def refine_watermark_area(reference_frame, selected_area):
    """Tighten a loose box when it contains a strong square logo contour.

    The refinement is deliberately conservative. If no high-confidence outer
    contour exists, the user's original selection is returned unchanged.
    """
    selected_area = clamp_area(selected_area, reference_frame.shape)
    if selected_area is None:
        return None
    ymin, ymax, xmin, xmax = selected_area
    crop = reference_frame[ymin:ymax, xmin:xmax]
    crop_height, crop_width = crop.shape[:2]
    if min(crop_height, crop_width) < 16:
        return selected_area
    tight_limit = max(96, int(round(min(reference_frame.shape[:2]) * 0.14)))
    if crop_height <= tight_limit and crop_width <= tight_limit:
        return selected_area

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    crop_area = float(crop_height * crop_width)
    minimum_side = max(12, int(round(min(crop_height, crop_width) * 0.35)))
    candidates = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < minimum_side or height < minimum_side:
            continue
        aspect_ratio = width / float(height)
        if not 0.7 <= aspect_ratio <= 1.3:
            continue
        box_area = float(width * height)
        area_ratio = box_area / crop_area
        if not 0.05 <= area_ratio <= 0.85:
            continue
        contour_area = abs(float(cv2.contourArea(contour)))
        fill_ratio = contour_area / max(1.0, box_area)
        if fill_ratio < 0.55:
            continue
        square_score = 1.0 - min(1.0, abs(1.0 - aspect_ratio))
        candidates.append((box_area * fill_ratio * square_score, x, y, width, height))

    if not candidates:
        return selected_area
    _, x, y, width, height = max(candidates, key=lambda item: item[0])
    refined = (
        ymin + y,
        ymin + y + height,
        xmin + x,
        xmin + x + width,
    )
    return clamp_area(refined, reference_frame.shape) or selected_area


def _structure_images(image):
    if image.ndim == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image
    gray_float = gray.astype(np.float32)
    high_pass = gray_float - cv2.GaussianBlur(gray_float, (0, 0), 3)
    laplacian = cv2.Laplacian(
        cv2.GaussianBlur(gray_float, (3, 3), 0),
        cv2.CV_32F,
        ksize=3,
    )
    edges = cv2.Canny(gray, 50, 150).astype(np.float32)
    return high_pass, laplacian, edges


def _peak_statistics(response, peak_location, template_size):
    peak_x, peak_y = peak_location
    template_width, template_height = template_size
    exclusion_x = max(3, template_width // 2)
    exclusion_y = max(3, template_height // 2)
    sidelobes = response.copy()
    y0 = max(0, peak_y - exclusion_y)
    y1 = min(response.shape[0], peak_y + exclusion_y + 1)
    x0 = max(0, peak_x - exclusion_x)
    x1 = min(response.shape[1], peak_x + exclusion_x + 1)
    sidelobes[y0:y1, x0:x1] = np.nan
    valid = sidelobes[np.isfinite(sidelobes)]
    if valid.size == 0:
        return float("inf"), -1.0
    peak_value = float(response[peak_y, peak_x])
    mean = float(np.mean(valid))
    std = float(np.std(valid))
    psr = (peak_value - mean) / max(std, 1e-6)
    second_peak = float(np.max(valid))
    return psr, second_peak


class MovingWatermarkTracker:
    """Track one watermark template locally, with global re-acquisition."""

    def __init__(
        self,
        template_image,
        initial_area,
        frame_shape,
        template_feature_frame=None,
        high_score=0.45,
        low_score=0.33,
        global_psr=7.0,
        peak_margin=0.08,
    ):
        self.frame_height, self.frame_width = frame_shape[:2]
        self.initial_area = clamp_area(initial_area, frame_shape)
        if self.initial_area is None:
            raise ValueError("Invalid moving watermark template area")
        ymin, ymax, xmin, xmax = self.initial_area
        target_width = xmax - xmin
        target_height = ymax - ymin
        if target_width < 8 or target_height < 8:
            raise ValueError("Moving watermark template is too small")
        if template_image.shape[1] != target_width or template_image.shape[0] != target_height:
            template_image = cv2.resize(template_image, (target_width, target_height))
        self.template_image = template_image
        if template_feature_frame is not None:
            if template_feature_frame.shape[:2] != (self.frame_height, self.frame_width):
                template_feature_frame = cv2.resize(
                    template_feature_frame,
                    (self.frame_width, self.frame_height),
                )
            feature_layers = _structure_images(template_feature_frame)
            self.template_high_pass = feature_layers[0][ymin:ymax, xmin:xmax]
            self.template_laplacian = feature_layers[1][ymin:ymax, xmin:xmax]
            self.template_edges = feature_layers[2][ymin:ymax, xmin:xmax]
        else:
            padded = cv2.copyMakeBorder(
                template_image,
                8,
                8,
                8,
                8,
                cv2.BORDER_REFLECT_101,
            )
            feature_layers = _structure_images(padded)
            self.template_high_pass = feature_layers[0][8:-8, 8:-8]
            self.template_laplacian = feature_layers[1][8:-8, 8:-8]
            self.template_edges = feature_layers[2][8:-8, 8:-8]
        if np.count_nonzero(self.template_edges) < 10:
            raise ValueError("Moving watermark template has insufficient structure")
        self.template_height, self.template_width = template_image.shape[:2]
        self.high_score = float(high_score)
        self.low_score = float(low_score)
        self.global_psr = float(global_psr)
        self.peak_margin = float(peak_margin)
        self.search_margin = max(
            24,
            int(round(max(self.template_width, self.template_height) * 0.6)),
        )
        self.last_area = None
        self.position_history = deque(maxlen=5)
        self.corner_anchors = self._build_corner_anchors()

    def _build_corner_anchors(self):
        ymin, ymax, xmin, xmax = self.initial_area
        mirrored_x = self.frame_width - xmax
        mirrored_y = self.frame_height - ymax
        anchors = [
            (ymin, ymin + self.template_height, xmin, xmin + self.template_width),
            (ymin, ymin + self.template_height, mirrored_x, mirrored_x + self.template_width),
            (mirrored_y, mirrored_y + self.template_height, xmin, xmin + self.template_width),
            (mirrored_y, mirrored_y + self.template_height, mirrored_x, mirrored_x + self.template_width),
        ]
        unique = []
        for area in anchors:
            area = clamp_area(area, (self.frame_height, self.frame_width))
            if area is not None and area not in unique:
                unique.append(area)
        return unique

    def _combined_response(self, search_image):
        search_high_pass, search_laplacian, search_edges = _structure_images(search_image)
        high_pass_response = cv2.matchTemplate(
            search_high_pass,
            self.template_high_pass,
            cv2.TM_CCOEFF_NORMED,
        )
        laplacian_response = cv2.matchTemplate(
            search_laplacian,
            self.template_laplacian,
            cv2.TM_CCOEFF_NORMED,
        )
        edge_response = cv2.matchTemplate(
            search_edges,
            self.template_edges,
            cv2.TM_CCOEFF_NORMED,
        )
        return high_pass_response * 0.30 + laplacian_response * 0.40 + edge_response * 0.30

    def _match_near(self, frame, expected_area):
        ymin, ymax, xmin, xmax = expected_area
        search_y0 = max(0, ymin - self.search_margin)
        search_x0 = max(0, xmin - self.search_margin)
        search_y1 = min(self.frame_height, ymax + self.search_margin)
        search_x1 = min(self.frame_width, xmax + self.search_margin)
        search = frame[search_y0:search_y1, search_x0:search_x1]
        if search.shape[0] < self.template_height or search.shape[1] < self.template_width:
            return None
        response = self._combined_response(search)
        _, score, _, location = cv2.minMaxLoc(response)
        x = search_x0 + location[0]
        y = search_y0 + location[1]
        return float(score), (y, y + self.template_height, x, x + self.template_width)

    def _global_match(self, frame):
        response = self._combined_response(frame)
        _, score, _, location = cv2.minMaxLoc(response)
        psr, second_peak = _peak_statistics(
            response,
            location,
            (self.template_width, self.template_height),
        )
        x, y = location
        area = (y, y + self.template_height, x, x + self.template_width)
        return float(score), area, float(psr), float(second_peak)

    @staticmethod
    def _center_distance(area_a, area_b):
        ay0, ay1, ax0, ax1 = area_a
        by0, by1, bx0, bx1 = area_b
        return float(np.hypot((ax0 + ax1 - bx0 - bx1) / 2, (ay0 + ay1 - by0 - by1) / 2))

    def _stabilize(self, area):
        if self.last_area is not None:
            jump_threshold = max(24.0, 0.75 * np.hypot(self.template_width, self.template_height))
            if self._center_distance(area, self.last_area) > jump_threshold:
                self.position_history.clear()
        self.position_history.append(area)
        positions = np.array(list(self.position_history), dtype=np.int32)
        median = np.median(positions, axis=0).round().astype(int)
        stabilized = tuple(map(int, median))
        self.last_area = stabilized
        return stabilized

    def locate(self, frame, force_global=False, allow_global=True):
        candidates = [self.last_area] if self.last_area is not None else list(self.corner_anchors)
        unique_candidates = []
        for area in candidates:
            if area not in unique_candidates:
                unique_candidates.append(area)

        local_matches = []
        if not force_global:
            for expected_area in unique_candidates:
                match = self._match_near(frame, expected_area)
                if match is not None:
                    local_matches.append(match)
            local_matches.sort(key=lambda item: item[0], reverse=True)
            if local_matches:
                best_score, best_area = local_matches[0]
                continuous = (
                    self.last_area is not None
                    and self._center_distance(best_area, self.last_area) <= self.search_margin
                )
                threshold = self.low_score if continuous else self.high_score
                if best_score >= threshold:
                    return self._stabilize(best_area), best_score

        if not allow_global:
            self.last_area = None
            self.position_history.clear()
            return None, local_matches[0][0] if local_matches else 0.0

        score, area, psr, second_peak = self._global_match(frame)
        if (
            score >= self.high_score
            and psr >= self.global_psr
            and score - second_peak >= self.peak_margin
        ):
            return self._stabilize(area), score
        self.last_area = None
        self.position_history.clear()
        return None, score


def smooth_tracking_results(areas, scores, fps=0.0):
    """Median-stabilize confident detections without inventing missing masks."""
    if not areas:
        return [], []
    smoothed = list(areas)
    window_radius = 2
    for index, area in enumerate(areas):
        if area is None:
            continue
        ymin, ymax, xmin, xmax = area
        width = xmax - xmin
        height = ymax - ymin
        nearby = []
        for neighbor_index in range(max(0, index - window_radius), min(len(areas), index + window_radius + 1)):
            neighbor = areas[neighbor_index]
            if neighbor is None:
                continue
            ny0, ny1, nx0, nx1 = neighbor
            distance = np.hypot(
                (xmin + xmax - nx0 - nx1) / 2,
                (ymin + ymax - ny0 - ny1) / 2,
            )
            if distance <= max(width, height) * 0.25:
                nearby.append(neighbor)
        if nearby:
            smoothed[index] = tuple(
                map(int, np.median(np.asarray(nearby, dtype=np.int32), axis=0).round())
            )

    return smoothed, scores
