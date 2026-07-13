import hashlib
import json
import os
import subprocess
import threading
import time
from collections import deque

import cv2
import numpy as np

from backend.scenedetect.detectors import ContentDetector
from backend.scenedetect.scene_manager import compute_downscale_factor
from backend.tools.common_tools import get_readable_path
from backend.tools.ffmpeg_cli import FFmpegCLI


MOVING_WATERMARK_PREPROCESS_SCHEMA = 1
MOVING_WATERMARK_TRACKER_VERSION = 2


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


def _file_signature(path):
    path = os.path.normcase(os.path.realpath(os.path.abspath(os.fspath(path))))
    stat = os.stat(path)
    return {
        "path": path,
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def canonicalize_ab_sections(ab_sections, frame_count):
    """Return merged inclusive frame intervals, or None for the full video."""
    if not ab_sections:
        return None
    frame_count = max(0, int(frame_count or 0))
    intervals = []
    for section in ab_sections:
        if isinstance(section, range):
            start, end = section.start, section.stop - 1
        elif isinstance(section, (tuple, list)) and len(section) >= 2:
            start, end = section[0], section[1]
        else:
            raise ValueError("Invalid A/B section in moving-watermark preprocessing")
        start, end = int(start), int(end)
        if frame_count:
            start = max(0, min(frame_count - 1, start))
            end = max(0, min(frame_count - 1, end))
        if end < start:
            continue
        intervals.append((start, end))
    intervals.sort()
    merged = []
    for start, end in intervals:
        if merged and start <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return tuple(merged)


def _frame_is_active(frame_no, canonical_ab_sections):
    if canonical_ab_sections is None:
        return True
    return any(start <= frame_no <= end for start, end in canonical_ab_sections)


def build_moving_watermark_preprocess_key(
    video_path,
    template_source_path,
    reference_frame_no,
    template_area,
    frame_shape,
    frame_count,
    fps,
    ab_sections=None,
    fallback_target_area=None,
):
    height, width = map(int, frame_shape[:2])
    area = template_area
    if isinstance(area, (list, tuple)) and len(area) == 1:
        area = area[0]
    normalized_area = None
    if isinstance(area, (list, tuple)) and len(area) == 4:
        normalized_area = [round(float(value), 8) for value in area]
    fallback_area = None
    if isinstance(fallback_target_area, (list, tuple)) and len(fallback_target_area) == 4:
        fallback_area = [int(value) for value in fallback_target_area]
    canonical_ab = canonicalize_ab_sections(ab_sections, frame_count)
    payload = {
        "schema": MOVING_WATERMARK_PREPROCESS_SCHEMA,
        "tracker_version": MOVING_WATERMARK_TRACKER_VERSION,
        "video": _file_signature(video_path),
        "template": _file_signature(template_source_path),
        "reference_frame_no": int(reference_frame_no or 0),
        "template_area": normalized_area,
        "fallback_target_area": fallback_area,
        "height": height,
        "width": width,
        "frame_count": int(frame_count or 0),
        "fps": round(float(fps or 0.0), 6),
        "ab_sections": canonical_ab,
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _build_tracker_for_preprocess(
    reference_frame,
    template_area,
    target_shape,
    fallback_target_area=None,
):
    source_height, source_width = reference_frame.shape[:2]
    area = template_area
    if isinstance(area, (list, tuple)) and len(area) == 1:
        area = area[0]
    if isinstance(area, (list, tuple)) and len(area) == 4:
        values = tuple(map(float, area))
        if all(0.0 <= value <= 1.0 for value in values):
            selected_area = (
                round(values[0] * source_height),
                round(values[1] * source_height),
                round(values[2] * source_width),
                round(values[3] * source_width),
            )
        else:
            selected_area = tuple(map(int, values))
    elif isinstance(fallback_target_area, (list, tuple)) and len(fallback_target_area) == 4:
        target_height, target_width = target_shape[:2]
        selected_area = (
            round(fallback_target_area[0] / max(1, target_height) * source_height),
            round(fallback_target_area[1] / max(1, target_height) * source_height),
            round(fallback_target_area[2] / max(1, target_width) * source_width),
            round(fallback_target_area[3] / max(1, target_width) * source_width),
        )
    else:
        raise ValueError("Moving watermark template is required")

    refined_area = refine_watermark_area(reference_frame, selected_area)
    if refined_area is None:
        raise ValueError("Moving watermark template is required")
    ymin, ymax, xmin, xmax = refined_area
    target_height, target_width = target_shape[:2]
    target_area = (
        round(ymin / source_height * target_height),
        round(ymax / source_height * target_height),
        round(xmin / source_width * target_width),
        round(xmax / source_width * target_width),
    )
    tracker = MovingWatermarkTracker(
        reference_frame[ymin:ymax, xmin:xmax].copy(),
        target_area,
        target_shape,
        template_feature_frame=reference_frame,
    )
    return tracker, tuple(map(int, selected_area)), tuple(map(int, refined_area)), target_area


def preprocess_moving_watermark(
    video_path,
    template_source_path,
    reference_frame_no,
    template_area,
    frame_shape,
    frame_count,
    fps,
    ab_sections=None,
    fallback_target_area=None,
    cancel_event=None,
):
    """Scan a video once for watermark tracking and scene boundaries.

    A previous implementation opened the same video in two OpenCV readers so
    tracking and scene detection could run in parallel.  Some Windows codec
    combinations can leave both readers waiting forever.  Feeding both
    detectors from this single decode pass is slightly more CPU-bound, but it
    removes that unbounded wait and also avoids decoding every frame twice.
    """
    started_at = time.perf_counter()
    height, width = map(int, frame_shape[:2])
    expected_count = int(frame_count or 0)
    artifact_key = build_moving_watermark_preprocess_key(
        video_path,
        template_source_path,
        reference_frame_no,
        template_area,
        (height, width),
        expected_count,
        fps,
        ab_sections,
        fallback_target_area,
    )
    reference_frame = read_video_frame(template_source_path, reference_frame_no, fps)
    if reference_frame is None:
        raise ValueError("Unable to read moving watermark reference frame")
    tracker, selected_area, refined_area, target_area = _build_tracker_for_preprocess(
        reference_frame,
        template_area,
        (height, width),
        fallback_target_area,
    )
    canonical_ab = canonicalize_ab_sections(ab_sections, expected_count)
    cap = cv2.VideoCapture(get_readable_path(video_path) or video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError("Unable to open video for moving watermark preprocessing")
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (actual_height, actual_width) != (height, width):
        cap.release()
        raise ValueError("Moving watermark preprocessing video dimensions changed")

    if cancel_event is not None and cancel_event.is_set():
        cap.release()
        raise RuntimeError("Moving watermark preprocessing cancelled")
    scene_detector = ContentDetector()
    scene_downscale = compute_downscale_factor(width)
    scene_starts = []
    areas = []
    scores = []
    frame_no = 0
    was_active = False
    try:
        while True:
            if cancel_event is not None and cancel_event.is_set():
                raise RuntimeError("Moving watermark preprocessing cancelled")
            ok, frame = cap.read()
            if not ok:
                break
            scene_frame = frame
            if scene_downscale > 1:
                scene_frame = cv2.resize(
                    frame,
                    (
                        round(frame.shape[1] / scene_downscale),
                        round(frame.shape[0] / scene_downscale),
                    ),
                    interpolation=cv2.INTER_LINEAR,
                )
            scene_starts.extend(scene_detector.process_frame(frame_no, scene_frame))
            if not _frame_is_active(frame_no, canonical_ab):
                areas.append(None)
                scores.append(0.0)
                tracker.last_area = None
                tracker.position_history.clear()
                was_active = False
                frame_no += 1
                continue
            force_global = not was_active
            allow_global = force_global or tracker.last_area is not None or frame_no % 12 == 0
            area, score = tracker.locate(
                frame,
                force_global=force_global,
                allow_global=allow_global,
            )
            areas.append(area)
            scores.append(float(score))
            was_active = True
            frame_no += 1
    finally:
        cap.release()

    if expected_count and frame_no != expected_count:
        raise ValueError(
            f"Moving watermark preprocessing decoded {frame_no} frames, expected {expected_count}"
        )
    areas, scores = smooth_tracking_results(areas, scores, fps)
    scene_starts.extend(scene_detector.post_process(frame_no))
    areas_array = np.full((frame_no, 4), -1, dtype=np.int32)
    for index, area in enumerate(areas):
        if area is not None:
            areas_array[index] = np.asarray(area, dtype=np.int32)
    scores_array = np.asarray(scores, dtype=np.float32)
    scene_array = np.asarray(sorted(set(scene_starts)), dtype=np.int32)
    return {
        "schema": MOVING_WATERMARK_PREPROCESS_SCHEMA,
        "key": artifact_key,
        "width": width,
        "height": height,
        "frame_count": frame_no,
        "fps": float(fps or 0.0),
        "areas": areas_array,
        "scores": scores_array,
        "scene_starts": scene_array,
        "selected_area": np.asarray(selected_area, dtype=np.int32),
        "refined_area": np.asarray(refined_area, dtype=np.int32),
        "target_area": np.asarray(target_area, dtype=np.int32),
        "detected_count": int(np.count_nonzero(areas_array[:, 0] >= 0)),
        "elapsed": float(time.perf_counter() - started_at),
        "canonical_ab": canonical_ab,
    }


def save_moving_watermark_preprocess(result, output_path):
    output_path = os.path.abspath(os.fspath(output_path))
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    temporary_path = f"{output_path}.{os.getpid()}.{threading.get_ident()}.tmp.npz"
    try:
        np.savez_compressed(
            temporary_path,
            schema=np.asarray(result["schema"], dtype=np.int32),
            key=np.asarray(result["key"]),
            width=np.asarray(result["width"], dtype=np.int32),
            height=np.asarray(result["height"], dtype=np.int32),
            frame_count=np.asarray(result["frame_count"], dtype=np.int64),
            fps=np.asarray(result["fps"], dtype=np.float64),
            areas=result["areas"],
            scores=result["scores"],
            scene_starts=result["scene_starts"],
            selected_area=result["selected_area"],
            refined_area=result["refined_area"],
            target_area=result["target_area"],
            detected_count=np.asarray(result["detected_count"], dtype=np.int64),
            elapsed=np.asarray(result["elapsed"], dtype=np.float64),
        )
        os.replace(temporary_path, output_path)
    finally:
        if os.path.exists(temporary_path):
            try:
                os.remove(temporary_path)
            except OSError:
                pass
    return output_path


def load_moving_watermark_preprocess(
    artifact_path,
    expected_key,
    frame_shape,
    frame_count,
    ab_sections=None,
    fps=None,
):
    height, width = map(int, frame_shape[:2])
    frame_count = int(frame_count or 0)
    with np.load(os.fspath(artifact_path), allow_pickle=False) as artifact:
        result = {name: artifact[name].copy() for name in artifact.files}
    schema = int(result["schema"].item())
    key = str(result["key"].item())
    stored_width = int(result["width"].item())
    stored_height = int(result["height"].item())
    stored_count = int(result["frame_count"].item())
    areas = result["areas"]
    scores = result["scores"]
    scene_starts = result["scene_starts"]
    stored_fps = float(result["fps"].item())
    if schema != MOVING_WATERMARK_PREPROCESS_SCHEMA or key != expected_key:
        raise ValueError("Moving watermark preprocessing key mismatch")
    if (stored_height, stored_width, stored_count) != (height, width, frame_count):
        raise ValueError("Moving watermark preprocessing video metadata mismatch")
    if fps is not None and abs(stored_fps - float(fps or 0.0)) > 1e-3:
        raise ValueError("Moving watermark preprocessing frame rate mismatch")
    if areas.dtype != np.int32 or areas.shape != (frame_count, 4):
        raise ValueError("Invalid moving watermark preprocessing areas")
    if scores.dtype != np.float32 or scores.shape != (frame_count,) or not np.all(np.isfinite(scores)):
        raise ValueError("Invalid moving watermark preprocessing scores")
    if np.any(scores < -1.001) or np.any(scores > 1.001):
        raise ValueError("Moving watermark preprocessing score is out of range")
    missing = np.all(areas == -1, axis=1)
    partial_missing = np.any(areas == -1, axis=1) & ~missing
    if np.any(partial_missing):
        raise ValueError("Invalid moving watermark preprocessing sentinel")
    valid = ~missing
    if np.any(valid):
        valid_areas = areas[valid]
        if np.any(valid_areas[:, 0] < 0) or np.any(valid_areas[:, 1] > height):
            raise ValueError("Moving watermark preprocessing area is out of bounds")
        if np.any(valid_areas[:, 2] < 0) or np.any(valid_areas[:, 3] > width):
            raise ValueError("Moving watermark preprocessing area is out of bounds")
        if np.any(valid_areas[:, 1] <= valid_areas[:, 0]) or np.any(valid_areas[:, 3] <= valid_areas[:, 2]):
            raise ValueError("Moving watermark preprocessing area is empty")
    if scene_starts.dtype != np.int32 or scene_starts.ndim != 1:
        raise ValueError("Invalid moving watermark preprocessing scene list")
    if scene_starts.size and (
        np.any(scene_starts < 0)
        or np.any(scene_starts >= frame_count)
        or not np.array_equal(scene_starts, np.unique(scene_starts))
    ):
        raise ValueError("Invalid moving watermark preprocessing scene frame")
    target_area = result["target_area"]
    if target_area.dtype != np.int32 or target_area.shape != (4,):
        raise ValueError("Invalid moving watermark preprocessing target area")
    target_height = int(target_area[1] - target_area[0])
    target_width = int(target_area[3] - target_area[2])
    if target_height <= 0 or target_width <= 0:
        raise ValueError("Invalid moving watermark preprocessing target size")
    if np.any(valid):
        valid_areas = areas[valid]
        if np.any(valid_areas[:, 1] - valid_areas[:, 0] != target_height):
            raise ValueError("Moving watermark preprocessing height changed")
        if np.any(valid_areas[:, 3] - valid_areas[:, 2] != target_width):
            raise ValueError("Moving watermark preprocessing width changed")
    if int(result["detected_count"].item()) != int(np.count_nonzero(valid)):
        raise ValueError("Moving watermark preprocessing detection count mismatch")
    for name in ("selected_area", "refined_area"):
        template_box = result[name]
        if template_box.dtype != np.int32 or template_box.shape != (4,):
            raise ValueError(f"Invalid moving watermark preprocessing {name}")
        if (
            template_box[0] < 0
            or template_box[2] < 0
            or template_box[1] <= template_box[0]
            or template_box[3] <= template_box[2]
        ):
            raise ValueError(f"Invalid moving watermark preprocessing {name}")
    canonical_ab = canonicalize_ab_sections(ab_sections, frame_count)
    if canonical_ab is not None:
        active = np.zeros(frame_count, dtype=bool)
        for start, end in canonical_ab:
            active[start:end + 1] = True
        if np.any(~active & valid) or np.any(scores[~active] != 0):
            raise ValueError("Moving watermark preprocessing violates A/B sections")
    result.update(
        schema=schema,
        key=key,
        width=stored_width,
        height=stored_height,
        frame_count=stored_count,
        fps=stored_fps,
        detected_count=int(result["detected_count"].item()),
        elapsed=float(result["elapsed"].item()),
    )
    return result


def preprocess_moving_watermark_to_file(output_path, **kwargs):
    expected_key = build_moving_watermark_preprocess_key(
        kwargs["video_path"],
        kwargs["template_source_path"],
        kwargs["reference_frame_no"],
        kwargs["template_area"],
        kwargs["frame_shape"],
        kwargs["frame_count"],
        kwargs["fps"],
        kwargs.get("ab_sections"),
        kwargs.get("fallback_target_area"),
    )
    if os.path.isfile(output_path):
        try:
            cached = load_moving_watermark_preprocess(
                output_path,
                expected_key,
                kwargs["frame_shape"],
                kwargs["frame_count"],
                kwargs.get("ab_sections"),
                kwargs.get("fps"),
            )
            return {
                "path": os.path.abspath(output_path),
                "key": expected_key,
                "frame_count": cached["frame_count"],
                "detected_count": cached["detected_count"],
                "elapsed": cached["elapsed"],
                "reused": True,
            }
        except Exception:
            pass
    result = preprocess_moving_watermark(**kwargs)
    save_moving_watermark_preprocess(result, output_path)
    return {
        "path": os.path.abspath(output_path),
        "key": result["key"],
        "frame_count": result["frame_count"],
        "detected_count": result["detected_count"],
        "elapsed": result["elapsed"],
        "reused": False,
    }
