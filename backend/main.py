import gc
import torch
import shutil
import traceback
import subprocess
import os
from pathlib import Path
import threading
import cv2
import sys
from functools import cached_property

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from backend.config import *
from backend.tools.hardware_accelerator import HardwareAccelerator
from backend.tools.common_tools import is_video_or_image, is_image_file, get_readable_path, read_image
from backend.inpaint.sttn_auto_inpaint import STTNAutoInpaint
from backend.inpaint.sttn_det_inpaint import STTNDetInpaint
from backend.inpaint.lama_inpaint import LamaInpaint
from backend.inpaint.opencv_inpaint import OpenCVInpaint
from backend.inpaint.propainter_inpaint import PropainterInpaint
from backend.tools.inpaint_tools import (
    build_fixed_watermark_masks,
    create_mask,
    batch_generator,
    expand_frame_ranges,
    is_frame_number_in_ab_sections,
)
from backend.tools.model_config import ModelConfig
from backend.tools.ffmpeg_cli import FFmpegCLI
from backend.tools.subtitle_detect import SubtitleDetect
from backend.tools.video_io import FramePrefetcher, FFmpegVideoWriter, create_processing_capture
from backend.tools.watermark_tracker import (
    MovingWatermarkTracker,
    build_moving_watermark_preprocess_key,
    load_moving_watermark_preprocess,
    preprocess_moving_watermark,
    read_video_frame,
    refine_watermark_area,
    smooth_tracking_results,
)
import tempfile
import multiprocessing
import time
from tqdm import tqdm
import numpy as np


_LAMA_INPAINT_CACHE = {}
_STTN_DET_INPAINT_CACHE = {}
_PROPAINTER_INPAINT_CACHE = {}
WATERMARK_INPAINT_MODES = (InpaintMode.FIXED_WATERMARK, InpaintMode.MOVING_WATERMARK)


def _device_cache_key(device):
    device_type = getattr(device, "type", None)
    device_index = getattr(device, "index", None)
    if device_type is not None:
        return f"{device_type}:{device_index}"
    return str(device)


def _preferred_audio_stream_spec(video_path, input_index=1):
    """Match FFmpeg's automatic audio choice: most channels, then first."""
    fallback = f"{input_index}:a:0?"
    try:
        import av

        with av.open(video_path) as container:
            audio_streams = list(container.streams.audio)
            if not audio_streams:
                return fallback

            def channel_count(stream):
                codec_context = stream.codec_context
                layout = getattr(codec_context, "layout", None)
                layout_channels = getattr(layout, "channels", ()) if layout is not None else ()
                try:
                    return max(
                        int(getattr(codec_context, "channels", 0) or 0),
                        len(layout_channels or ()),
                    )
                except (TypeError, ValueError):
                    return 0

            selected = max(
                audio_streams,
                key=lambda stream: (channel_count(stream), -int(stream.index)),
            )
            return f"{input_index}:{int(selected.index)}?"
    except Exception:
        return fallback

class SubtitleRemover:
    def __init__(self, vd_path, gui_mode=False, video_bitrate_mbps=None):
        # 线程锁
        self.lock = threading.RLock()
        # 用户指定的字幕区域位置
        self.sub_areas = []
        # 是否为gui运行，gui运行需要显示预览
        self.gui_mode = gui_mode
        self.hardware_accelerator = HardwareAccelerator.instance()
        # 是否使用硬件加速
        self.hardware_accelerator.set_enabled(config.hardwareAcceleration.value)
        self.model_config = ModelConfig()
        # 判断是否为图片
        self.is_picture = is_image_file(str(vd_path))
        # 视频路径
        self.video_path = vd_path
        self.video_cap = cv2.VideoCapture(get_readable_path(vd_path))
        # 通过视频路径获取视频名称
        self.vd_name = Path(self.video_path).stem
        # 视频帧总数
        self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT) + 0.5)
        # 视频帧率
        self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
        # 视频尺寸
        self.size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.mask_size = (int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT)), int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH)))
        self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        # 创建视频临时对象，windows下delete=True会有permission denied的报错
        self.video_temp_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
        # 创建视频写对象（使用 FFmpeg libx264 编码，比 mp4v 质量更好、文件更小）
        try:
            self.video_writer = FFmpegVideoWriter(
                get_readable_path(self.video_temp_file.name),
                self.fps,
                self.size,
                bitrate_mbps=(
                    config.videoOutputBitrateMbps.value
                    if video_bitrate_mbps is None
                    else video_bitrate_mbps
                ),
            )
        except Exception:
            self.video_writer = cv2.VideoWriter(get_readable_path(self.video_temp_file.name), cv2.VideoWriter_fourcc(*'mp4v'), self.fps, self.size)
        self.video_out_path = os.path.abspath(os.path.join(os.path.dirname(self.video_path), f'{self.vd_name}_no_sub.mp4'))
        self.propainter_inpaint = None
        self.ext = os.path.splitext(vd_path)[-1]
        if self.is_picture:
            pic_dir = os.path.join(os.path.dirname(self.video_path), 'no_sub')
            if not os.path.exists(pic_dir):
                os.makedirs(pic_dir)
            self.video_out_path = os.path.join(pic_dir, f'{self.vd_name}{self.ext}')

        # 总处理进度
        self.progress_total = 0
        self.progress_remover = 0
        self.isFinished = False
        # 是否将原音频嵌入到去除字幕后的视频
        self.is_successful_merged = False
        # 进度监听器列表
        self.progress_listeners = []
        # inpaint的frame_no区域列表, 默认为inpaint所有帧
        self.ab_sections = None
        self.subtitle_intervals = None
        self.tracking_reference_frame_no = 0
        self.tracking_template_area = None
        self.tracking_template_source_path = None
        self.moving_watermark_preprocess_path = None
        # Snapshot performance-related options on the job instance.  The GUI
        # worker process is intentionally long-lived, so reading the child
        # process' global config here would make later UI changes ineffective.
        self.moving_watermark_fast_mode = bool(config.movingWatermarkFastMode.value)
        self.propainter_max_load_num = int(config.propainterMaxLoadNum.value)
        self.report_processing_phase = lambda *args: None
        self.preview_emit_interval = 0.2 if gui_mode else 0.0
        self._last_preview_emit_time = 0.0

    @staticmethod
    def is_current_frame_no_start(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no == frame_no:
                return True
        return False

    @staticmethod
    def find_frame_no_end(frame_no, continuous_frame_no_list):
        """
        判断给定的帧号是否为开头，是的话返回结束帧号，不是的话返回-1
        """
        for start_no, end_no in continuous_frame_no_list:
            if start_no <= frame_no <= end_no:
                return end_no
        return -1

    @staticmethod
    def intersect_intervals(primary_intervals, secondary_intervals):
        def normalize(section):
            if isinstance(section, range):
                return section.start, section.stop - 1
            return section[0], section[1]
        if not primary_intervals:
            return secondary_intervals
        if not secondary_intervals:
            return primary_intervals
        result = []
        for section_a in primary_intervals:
            start_a, end_a = normalize(section_a)
            for section_b in secondary_intervals:
                start_b, end_b = normalize(section_b)
                start = max(start_a, start_b)
                end = min(end_a, end_b)
                if start <= end:
                    result.append((start, end))
        return result

    @staticmethod
    def count_interval_frames(intervals):
        if not intervals:
            return 0
        total = 0
        for section in intervals:
            if isinstance(section, range):
                total += max(0, section.stop - section.start)
            else:
                total += max(0, int(section[1]) - int(section[0]) + 1)
        return total

    def update_progress(self, tbar, increment):
        tbar.update(increment)
        current_percentage = (tbar.n / tbar.total) * 100
        progress = int(current_percentage)
        self.progress_remover = progress
        if progress != self.progress_total:
            self.progress_total = progress
            self.notify_progress_listeners()

    def append_output(self, *args):
        """输出信息到控制台
        Args:
            *args: 要输出的内容，多个参数将用空格连接
        """
        print(*args)
    
    def add_progress_listener(self, listener):
        """
        添加进度监听器
        
        Args:
            listener: 一个回调函数，接收参数 (progress_total, isFinished)
        """
        if listener not in self.progress_listeners:
            self.progress_listeners.append(listener)
    
    def remove_progress_listener(self, listener):
        """
        移除进度监听器
        
        Args:
            listener: 要移除的监听器函数
        """
        if listener in self.progress_listeners:
            self.progress_listeners.remove(listener)
            
    def notify_progress_listeners(self):
        """
        通知所有进度监听器当前进度
        """
        for listener in self.progress_listeners:
            try:
                listener(self.progress_total, self.isFinished)
            except Exception as e:
                traceback.print_exc()

    def update_preview_with_comp(self, frame_ori, frame_comp):
        """
        更新预览
        """
        pass

    def push_preview_with_comp(self, frame_ori, frame_comp, force=False):
        if not self.gui_mode:
            return
        now = time.monotonic()
        if force or now - self._last_preview_emit_time >= self.preview_emit_interval:
            self._last_preview_emit_time = now
            if callable(frame_ori):
                frame_ori = frame_ori()
            if callable(frame_comp):
                frame_comp = frame_comp()
            self.update_preview_with_comp(frame_ori, frame_comp)

    def propainter_mode(self, tbar):
        sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
        sub_list = sub_detector.find_subtitle_frame_no(sub_remover=self)
        if len(sub_list) == 0:
            raise Exception(tr['Main']['NoSubtitleDetected'].format(self.video_path))
        continuous_frame_no_list = sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        scene_div_points = sub_detector.get_scene_div_frame_no(self.video_path)
        continuous_frame_no_list = sub_detector.split_range_by_scene(continuous_frame_no_list,
                                                                          scene_div_points)
        del sub_detector
        gc.collect()        
        propainter_inpaint = self.propainter_inpaint_model
        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        index = 0
        # 使用帧预读取，I/O 与推理重叠
        read_cap = create_processing_capture(
            self.video_path,
            self.frame_width,
            self.frame_height,
            fps=self.fps,
            frame_count=self.frame_count,
            fallback_cap=self.video_cap,
        )
        reader = FramePrefetcher(read_cap)
        while True:
            ret, frame = reader.read()
            if not ret:
                break
            index += 1
            # 如果当前帧没有水印/文本则直接写
            if index not in sub_list.keys():
                self.video_writer.write(frame)
                # self.append_output(f'write frame: {index}')
                self.update_progress(tbar, increment=1)
                self.push_preview_with_comp(frame, frame)
                continue
            # 如果有水印，判断该帧是不是开头帧
            else:
                # 如果是开头帧，则批推理到尾帧
                if self.is_current_frame_no_start(index, continuous_frame_no_list):
                    # self.append_output(f'No 1 Current index: {index}')
                    start_frame_no = index
                    # self.append_output(f'find start: {start_frame_no}')
                    # 找到结束帧
                    end_frame_no = self.find_frame_no_end(index, continuous_frame_no_list)
                    # 判断当前帧号是不是字幕起始位置
                    # 如果获取的结束帧号不为-1则说明
                    if end_frame_no != -1:
                        # self.append_output(f'find end: {end_frame_no}')
                        # ************ 读取该区间所有帧 start ************
                        temp_frames = list()
                        # 将头帧加入处理列表
                        temp_frames.append(frame)
                        inner_index = 0
                        # 一直读取到尾帧
                        while index < end_frame_no:
                            ret, frame = reader.read()
                            if not ret:
                                break
                            index += 1
                            temp_frames.append(frame)
                        # ************ 读取该区间所有帧 end ************
                        if len(temp_frames) < 1:
                            # 没有待处理，直接跳过
                            continue
                        elif len(temp_frames) == 1:
                            inner_index += 1
                            single_mask = create_mask(self.mask_size, sub_list[index])
                            inpainted_frame = self.lama_inpaint.inpaint(frame, single_mask)
                            self.video_writer.write(inpainted_frame)
                            # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                            self.update_progress(tbar, increment=1)
                            continue
                        else:
                            # 将读取的视频帧分批处理
                            # 1. 获取当前批次使用的mask
                            mask = create_mask(self.mask_size, sub_list[start_frame_no])
                            for batch in batch_generator(temp_frames, self.propainter_max_load_num):
                                # 2. 调用批推理
                                if len(batch) == 1:
                                    single_mask = create_mask(self.mask_size, sub_list[start_frame_no])
                                    inpainted_frame = self.lama_inpaint.inpaint(frame, single_mask)
                                    self.video_writer.write(inpainted_frame)
                                    # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[start_frame_no]}')
                                    inner_index += 1
                                    self.update_progress(tbar, increment=1)
                                elif len(batch) > 1:
                                    inpainted_frames = propainter_inpaint(batch, mask)
                                    for i, inpainted_frame in enumerate(inpainted_frames):
                                        self.video_writer.write(inpainted_frame)
                                        # self.append_output(f'write frame: {start_frame_no + inner_index} with mask {sub_list[index]}')
                                        inner_index += 1
                                        self.push_preview_with_comp(
                                            lambda source=batch[i], preview_mask=mask: np.clip(
                                                source + preview_mask[:, :, np.newaxis] * 0.3,
                                                0,
                                                255,
                                            ).astype(np.uint8),
                                            inpainted_frame,
                                        )
                                self.update_progress(tbar, increment=len(batch))
        reader.release()

    def _create_fixed_watermark_mask(self, areas=None):
        """Create a full-frame mask directly from the user-selected regions."""
        mask_area_coordinates = []
        for sub_area in (self.sub_areas if areas is None else areas):
            ymin, ymax, xmin, xmax = sub_area
            mask_area_coordinates.append((xmin, xmax, ymin, ymax))
        # Fixed selections must stay tight. Subtitle masks deliberately add a
        # 10px deviation, which produces a large rectangular patch for logos.
        return create_mask(self.mask_size, mask_area_coordinates, deviation=0)

    @staticmethod
    def _fixed_watermark_preview(frame, mask):
        """Show the selected boundary without covering the source with gray."""
        preview = frame.copy()
        contours, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(preview, contours, -1, (0, 255, 0), 2)
        return preview

    @staticmethod
    def _blend_fixed_watermark_frame(original, completed, selection_mask):
        """Keep the selection fully erased and feather only its clean outer ring."""
        _, outer_mask, alpha = build_fixed_watermark_masks(selection_mask)
        alpha = alpha[:, :, np.newaxis]
        blended = completed.astype(np.float32) * alpha + original.astype(np.float32) * (1.0 - alpha)
        return np.clip(blended, 0, 255).astype(np.uint8), outer_mask

    def _inpaint_fixed_watermark_batch(
        self,
        frames,
        mask=None,
        fast=False,
        moving_areas=None,
    ):
        """Inpaint a contiguous frame batch while handling a one-frame tail."""
        if moving_areas is not None and len(moving_areas) != len(frames):
            raise ValueError(
                f"Expected {len(frames)} moving-watermark areas, got {len(moving_areas)}"
            )
        if len(frames) == 1:
            if moving_areas is not None:
                frame_mask = self._create_fixed_watermark_mask([moving_areas[0]])
            else:
                frame_mask = mask[0] if isinstance(mask, (list, tuple)) else mask
            _, outer_mask, _ = build_fixed_watermark_masks(frame_mask)
            rgb_frame = cv2.cvtColor(frames[0], cv2.COLOR_BGR2RGB)
            completed_rgb = self.lama_inpaint.inpaint(rgb_frame, outer_mask)
            completed = cv2.cvtColor(completed_rgb, cv2.COLOR_RGB2BGR)
            blended, _ = self._blend_fixed_watermark_frame(frames[0], completed, frame_mask)
            return [blended]
        return self.propainter_inpaint_model.inpaint_fixed_watermark(
            frames,
            mask,
            moving_areas=moving_areas,
            raft_iter=(
                8
                if fast and self.moving_watermark_fast_mode
                else None
            ),
        )

    def fixed_watermark_mode(self, tbar):
        """Remove a fixed watermark from manually selected regions without OCR."""
        mask = self._create_fixed_watermark_mask()
        if not np.any(mask):
            raise ValueError(tr['Main']['FixedWatermarkAreaRequired'])

        self.append_output(tr['Main']['ProcessingStartRemovingFixedWatermark'])
        self.append_output(tr['Main']['FixedWatermarkStaticOnlyNote'])
        max_batch_size = max(2, int(self.propainter_max_load_num))
        overlap = min(10, max(0, max_batch_size // 5))
        if max_batch_size - overlap < 2:
            overlap = 0

        try:
            scene_starts = {
                max(0, frame_number - 1)
                for frame_number in SubtitleDetect.get_scene_div_frame_no(self.video_path)
            }
        except Exception:
            scene_starts = set()

        read_cap = create_processing_capture(
            self.video_path,
            self.frame_width,
            self.frame_height,
            fps=self.fps,
            frame_count=self.frame_count,
            fallback_cap=self.video_cap,
        )
        reader = FramePrefetcher(read_cap)
        pending_frames = []

        def write_processed_batch(frames, keep_overlap=False):
            if not frames:
                return []
            inpainted_frames = self._inpaint_fixed_watermark_batch(frames, mask)
            keep_count = overlap if keep_overlap and len(frames) > overlap else 0
            write_count = len(frames) - keep_count
            for index in range(write_count):
                inpainted_frame = inpainted_frames[index]
                self.video_writer.write(inpainted_frame)
                self.push_preview_with_comp(
                    lambda source=frames[index], preview_mask=mask: self._fixed_watermark_preview(
                        source,
                        preview_mask,
                    ),
                    inpainted_frame,
                )
            self.update_progress(tbar, increment=write_count)
            return frames[write_count:]

        frame_no = 0
        try:
            while True:
                ret, frame = reader.read()
                if not ret:
                    break

                if frame_no in scene_starts and pending_frames:
                    pending_frames = write_processed_batch(pending_frames)

                if is_frame_number_in_ab_sections(frame_no, self.ab_sections):
                    pending_frames.append(frame)
                    if len(pending_frames) >= max_batch_size:
                        pending_frames = write_processed_batch(pending_frames, keep_overlap=True)
                else:
                    if pending_frames:
                        pending_frames = write_processed_batch(pending_frames)
                    self.video_writer.write(frame)
                    self.update_progress(tbar, increment=1)
                    self.push_preview_with_comp(frame, frame)
                frame_no += 1

            if pending_frames:
                write_processed_batch(pending_frames)
        finally:
            reader.release()

    def _create_moving_watermark_tracker(self):
        if len(self.sub_areas) != 1:
            raise ValueError(tr['Main']['MovingWatermarkTemplateRequired'])

        template_source = self.tracking_template_source_path or self.video_path
        if self.tracking_template_source_path and not os.path.isfile(template_source):
            raise ValueError(tr['Main']['MovingWatermarkReferenceReadFailed'])
        reference_frame = read_video_frame(
            template_source,
            self.tracking_reference_frame_no,
            self.fps,
        )
        if reference_frame is None:
            raise ValueError(tr['Main']['MovingWatermarkReferenceReadFailed'])

        source_height, source_width = reference_frame.shape[:2]
        template_area = self.tracking_template_area
        if isinstance(template_area, (list, tuple)) and len(template_area) == 1:
            template_area = template_area[0]
        if isinstance(template_area, (list, tuple)) and len(template_area) == 4:
            values = tuple(map(float, template_area))
            if all(0.0 <= value <= 1.0 for value in values):
                selected_area = (
                    round(values[0] * source_height),
                    round(values[1] * source_height),
                    round(values[2] * source_width),
                    round(values[3] * source_width),
                )
            else:
                selected_area = tuple(map(int, values))
        else:
            target_area = self.sub_areas[0]
            selected_area = (
                round(target_area[0] / max(1, self.frame_height) * source_height),
                round(target_area[1] / max(1, self.frame_height) * source_height),
                round(target_area[2] / max(1, self.frame_width) * source_width),
                round(target_area[3] / max(1, self.frame_width) * source_width),
            )

        refined_area = refine_watermark_area(reference_frame, selected_area)
        if refined_area is None:
            raise ValueError(tr['Main']['MovingWatermarkTemplateRequired'])
        if tuple(map(int, selected_area)) != tuple(map(int, refined_area)):
            self.append_output(
                tr['Main']['MovingWatermarkTemplateRefined'].format(selected_area, refined_area)
            )

        ymin, ymax, xmin, xmax = refined_area
        template_image = reference_frame[ymin:ymax, xmin:xmax].copy()
        normalized_refined_area = (
            ymin / source_height,
            ymax / source_height,
            xmin / source_width,
            xmax / source_width,
        )
        target_area = (
            round(normalized_refined_area[0] * self.frame_height),
            round(normalized_refined_area[1] * self.frame_height),
            round(normalized_refined_area[2] * self.frame_width),
            round(normalized_refined_area[3] * self.frame_width),
        )
        tracker = MovingWatermarkTracker(
            template_image,
            target_area,
            self.mask_size,
            template_feature_frame=reference_frame,
        )
        return tracker, target_area

    def _scan_moving_watermark(self, tracker):
        self.append_output(tr['Main']['ProcessingStartTrackingWatermark'])
        started_at = time.time()
        read_cap = create_processing_capture(
            self.video_path,
            self.frame_width,
            self.frame_height,
            fps=self.fps,
            frame_count=self.frame_count,
            prefer_cuda=False,
        )
        reader = FramePrefetcher(read_cap)
        areas = []
        scores = []
        frame_no = 0
        was_active = False
        try:
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                active = is_frame_number_in_ab_sections(frame_no, self.ab_sections)
                if not active:
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
            reader.release()

        areas, scores = smooth_tracking_results(areas, scores, self.fps)
        detected_count = sum(area is not None for area in areas)
        self.append_output(
            tr['Main']['MovingWatermarkTrackingSummary'].format(
                detected_count,
                len(areas),
                time.time() - started_at,
            )
        )
        return areas, scores

    @staticmethod
    def _moving_batch_needs_flush(pending_areas, next_area):
        if not pending_areas:
            return False
        previous = pending_areas[-1]
        py0, py1, px0, px1 = previous
        ny0, ny1, nx0, nx1 = next_area
        template_width = max(1, nx1 - nx0)
        template_height = max(1, ny1 - ny0)
        center_distance = np.hypot(
            (px0 + px1 - nx0 - nx1) / 2,
            (py0 + py1 - ny0 - ny1) / 2,
        )
        if center_distance > max(24.0, 0.75 * np.hypot(template_width, template_height)):
            return True

        all_areas = pending_areas + [next_area]
        union_width = max(area[3] for area in all_areas) - min(area[2] for area in all_areas)
        union_height = max(area[1] for area in all_areas) - min(area[0] for area in all_areas)
        return union_width > template_width * 2.5 or union_height > template_height * 2.5

    def moving_watermark_mode(self, tbar):
        """Track one watermark template and inpaint only confidently located frames."""
        template_source = self.tracking_template_source_path or self.video_path
        if self.tracking_template_source_path and not os.path.isfile(template_source):
            raise ValueError(tr['Main']['MovingWatermarkReferenceReadFailed'])
        expected_key = build_moving_watermark_preprocess_key(
            self.video_path,
            template_source,
            self.tracking_reference_frame_no,
            self.tracking_template_area,
            self.mask_size,
            self.frame_count,
            self.fps,
            self.ab_sections,
            self.sub_areas[0],
        )
        preprocess_result = None
        if self.moving_watermark_preprocess_path:
            try:
                preprocess_result = load_moving_watermark_preprocess(
                    self.moving_watermark_preprocess_path,
                    expected_key,
                    self.mask_size,
                    self.frame_count,
                    self.ab_sections,
                    self.fps,
                )
                self.append_output(tr['Main']['MovingWatermarkPreprocessUsing'])
            except Exception as error:
                self.append_output(
                    tr['Main']['MovingWatermarkPreprocessInvalid'].format(error)
                )
        if preprocess_result is None:
            self.append_output(tr['Main']['ProcessingStartTrackingWatermark'])
            preprocess_result = preprocess_moving_watermark(
                self.video_path,
                template_source,
                self.tracking_reference_frame_no,
                self.tracking_template_area,
                self.mask_size,
                self.frame_count,
                self.fps,
                self.ab_sections,
                self.sub_areas[0],
            )
        selected_area = tuple(map(int, preprocess_result['selected_area']))
        refined_area = tuple(map(int, preprocess_result['refined_area']))
        if selected_area != refined_area:
            self.append_output(
                tr['Main']['MovingWatermarkTemplateRefined'].format(
                    selected_area,
                    refined_area,
                )
            )
        tracked_area_array = preprocess_result['areas']
        tracked_areas = [
            None if row[0] < 0 else tuple(map(int, row))
            for row in tracked_area_array
        ]
        scene_starts = set(map(int, preprocess_result['scene_starts']))
        self.append_output(
            tr['Main']['MovingWatermarkTrackingSummary'].format(
                preprocess_result['detected_count'],
                preprocess_result['frame_count'],
                preprocess_result['elapsed'],
            )
        )
        self.report_processing_phase("inpaint_started", self.video_path)
        self.append_output(tr['Main']['ProcessingStartRemovingMovingWatermark'])
        self.append_output(
            f"[Info] Moving-watermark RAFT iterations: "
            f"{8 if self.moving_watermark_fast_mode else 20}"
        )

        max_batch_size = max(2, int(self.propainter_max_load_num))
        overlap = min(10, max(0, max_batch_size // 5))
        if max_batch_size - overlap < 2:
            overlap = 0

        read_cap = create_processing_capture(
            self.video_path,
            self.frame_width,
            self.frame_height,
            fps=self.fps,
            frame_count=self.frame_count,
            fallback_cap=self.video_cap,
            prefer_cuda=False,
        )
        reader = FramePrefetcher(read_cap)
        pending_frames = []
        pending_areas = []

        def write_processed_batch(keep_overlap=False):
            nonlocal pending_frames, pending_areas
            if not pending_frames:
                return
            completed_frames = self._inpaint_fixed_watermark_batch(
                pending_frames,
                fast=True,
                moving_areas=pending_areas,
            )
            keep_count = overlap if keep_overlap and len(pending_frames) > overlap else 0
            write_count = len(pending_frames) - keep_count
            for index in range(write_count):
                self.video_writer.write(completed_frames[index])
                self.push_preview_with_comp(
                    lambda source=pending_frames[index], area=pending_areas[index]: self._fixed_watermark_preview(
                        source,
                        self._create_fixed_watermark_mask([area]),
                    ),
                    completed_frames[index],
                )
            self.update_progress(tbar, increment=write_count)
            if keep_count:
                pending_frames = pending_frames[write_count:]
                pending_areas = pending_areas[write_count:]
            else:
                pending_frames = []
                pending_areas = []

        frame_no = 0
        try:
            while True:
                ok, frame = reader.read()
                if not ok:
                    break
                if frame_no >= len(tracked_areas):
                    raise ValueError("Video contains more frames than moving watermark preprocessing")
                area = tracked_areas[frame_no]

                if frame_no in scene_starts and pending_frames:
                    write_processed_batch()

                if area is None:
                    write_processed_batch()
                    self.video_writer.write(frame)
                    self.update_progress(tbar, increment=1)
                    self.push_preview_with_comp(frame, frame)
                    frame_no += 1
                    continue

                if self._moving_batch_needs_flush(pending_areas, area):
                    write_processed_batch()
                pending_frames.append(frame)
                pending_areas.append(area)
                if len(pending_frames) >= max_batch_size:
                    write_processed_batch(keep_overlap=True)
                frame_no += 1

            if frame_no != len(tracked_areas):
                raise ValueError(
                    f"Video decoded {frame_no} frames, expected {len(tracked_areas)}"
                )
            write_processed_batch()
        finally:
            reader.release()

    def sttn_auto_mode(self, tbar):
        """
        使用sttn对选中区域进行重绘，不进行字幕检测
        """
        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        mask_area_coordinates = []
        for sub_area in self.sub_areas:
            ymin, ymax, xmin, xmax = sub_area
            mask_area_coordinates.append((xmin, xmax, ymin, ymax))
        mask = create_mask(self.mask_size, mask_area_coordinates)
        sttn_video_inpaint = STTNAutoInpaint(self.hardware_accelerator.device, self.model_config.STTN_AUTO_MODEL_PATH, self.video_path)
        original_ab_sections = self.ab_sections
        try:
            if self.subtitle_intervals:
                if original_ab_sections:
                    self.ab_sections = self.intersect_intervals(original_ab_sections, self.subtitle_intervals)
                else:
                    self.ab_sections = self.subtitle_intervals
                active_frame_count = self.count_interval_frames(self.ab_sections)
                skipped_frame_count = max(0, self.frame_count - active_frame_count)
                skipped_ratio = (skipped_frame_count / self.frame_count * 100) if self.frame_count else 0
                self.append_output(f"Subtitle intervals: {self.ab_sections}")
                self.append_output(
                    f"Skip non-subtitle frames: {skipped_frame_count}/{self.frame_count} ({skipped_ratio:.1f}%)"
                )
            sttn_video_inpaint(input_mask=mask, input_sub_remover=self, tbar=tbar)
        finally:
            self.ab_sections = original_ab_sections

    def video_inpaint(self, tbar, model):
        sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
        sub_list = sub_detector.find_subtitle_frame_no(sub_remover=self)
        if len(sub_list) == 0:
            raise Exception(tr['Main']['NoSubtitleDetected'].format(self.video_path))
        continuous_frame_no_list = sub_detector.find_continuous_ranges_with_same_mask(sub_list)
        tbar.write(f"Subtitle detected: {continuous_frame_no_list}")
        continuous_frame_no_list = expand_frame_ranges(continuous_frame_no_list, config.subtitleTimelineBackwardFrameCount.value, config.subtitleTimelineForwardFrameCount.value)
        tbar.write(f"Subtitle timeline expand ({config.subtitleTimelineBackwardFrameCount.value} <- -> {config.subtitleTimelineForwardFrameCount.value}): {continuous_frame_no_list}")
        continuous_frame_no_list = sub_detector.filter_and_merge_intervals(continuous_frame_no_list, config.sttnReferenceLength.value)
        tbar.write(f'Subtitle filter_and_merge_intervals: {continuous_frame_no_list}')
        del sub_detector
        gc.collect()
        start_end_map = dict()
        for start, end in continuous_frame_no_list:
            # 确保区间不超出视频总帧数，否则会导致 FramePrefetcher 哨兵被内循环消费后外层死锁
            start_end_map[start] = min(end, self.frame_count)
        current_frame_index = 0
        self.append_output(tr['Main']['ProcessingStartRemovingSubtitles'])
        # 使用帧预读取，I/O 与推理重叠
        read_cap = create_processing_capture(
            self.video_path,
            self.frame_width,
            self.frame_height,
            fps=self.fps,
            frame_count=self.frame_count,
            fallback_cap=self.video_cap,
        )
        reader = FramePrefetcher(read_cap)
        while True:
            ret, frame = reader.read()
            # 如果读取到为，则结束
            if not ret:
                break
            current_frame_index += 1
            # 判断当前帧号是不是字幕区间开始, 如果不是，则直接写
            if current_frame_index not in start_end_map.keys():
                self.video_writer.write(frame)
                # self.append_output(f'write frame: {current_frame_index}')
                self.update_progress(tbar, increment=1)
                self.push_preview_with_comp(frame, frame)
            # 如果是区间开始，则找到尾巴
            else:
                start_frame_index = current_frame_index
                end_frame_index = start_end_map[current_frame_index]
                tbar.write(f'processing frame {start_frame_index} to {end_frame_index}')
                # 用于存储需要去字幕的视频帧
                frames_need_inpaint = list()
                frames_need_inpaint.append(frame)
                inner_index = 0
                # 接着往下读，直到读取到尾巴
                for j in range(end_frame_index - start_frame_index):
                    ret, frame = reader.read()
                    if not ret:
                        break
                    current_frame_index += 1
                    frames_need_inpaint.append(frame)
                mask_area_coordinates = []
                # 1. 获取当前批次的mask坐标全集
                for mask_index in range(start_frame_index, end_frame_index):
                    if mask_index in sub_list.keys():
                        for area in sub_list[mask_index]:
                            xmin, xmax, ymin, ymax = area
                            # 判断是不是非字幕区域(如果宽大于长，则认为是错误检测)
                            if (ymax - ymin) - (xmax - xmin) > config.subtitleYXAxisDifferencePixel.value:
                                continue
                            if area not in mask_area_coordinates:
                                mask_area_coordinates.append(area)
                # 1. 获取当前批次使用的mask
                mask = create_mask(self.mask_size, mask_area_coordinates)
                # self.append_output(f'inpaint with mask: {mask_area_coordinates}')
                for batch in batch_generator(frames_need_inpaint, config.getSttnMaxLoadNum()):
                    # 2. 调用批推理
                    if len(batch) >= 1:
                        inpainted_frames = model(batch, mask)
                        for i, inpainted_frame in enumerate(inpainted_frames):
                            self.video_writer.write(inpainted_frame)
                            # self.append_output(f'write frame: {start_frame_index + inner_index} with mask')
                            inner_index += 1
                            self.push_preview_with_comp(
                                lambda source=batch[i], preview_mask=mask: np.clip(
                                    source + preview_mask[:, :, np.newaxis] * 0.3,
                                    0,
                                    255,
                                ).astype(np.uint8),
                                inpainted_frame,
                            )
                    self.update_progress(tbar, increment=len(batch))
        reader.release()

    def run(self):
        """Run one job and deterministically release per-job media resources."""
        completed = False
        try:
            result = self._run_job()
            completed = bool(self.isFinished)
            return result
        finally:
            try:
                self.video_cap.release()
            except Exception:
                pass
            try:
                self.video_writer.release()
            except Exception:
                # Preserve the original processing error. On an otherwise
                # successful path, writer finalization failures must surface.
                if completed:
                    raise
            try:
                self.video_temp_file.close()
            except Exception:
                pass
            try:
                if os.path.exists(self.video_temp_file.name):
                    os.remove(self.video_temp_file.name)
            except OSError:
                pass

    def _run_job(self):
        # 记录开始时间
        start_time = time.time()
        if self.is_picture and config.inpaintMode.value == InpaintMode.MOVING_WATERMARK:
            raise ValueError(tr['Main']['MovingWatermarkTemplateRequired'])
        if len(self.sub_areas) == 0:
            if config.inpaintMode.value == InpaintMode.MOVING_WATERMARK:
                raise ValueError(tr['Main']['MovingWatermarkTemplateRequired'])
            if config.inpaintMode.value == InpaintMode.FIXED_WATERMARK:
                raise ValueError(tr['Main']['FixedWatermarkAreaRequired'])
            self.append_output(tr['Main']['FullScreenProcessingNote'])
            self.sub_areas.append((0, self.frame_height, 0, self.frame_width))
        self.append_output(tr['Main']['SubtitleArea'].format(self.sub_areas))
        self.append_output(tr['Main']['ABSection'].format(str(self.ab_sections).replace("range", "") if self.ab_sections is not None and len(self.ab_sections) > 0 else tr['Main']['ABSectionAll']))
        # 如果使用GPU加速，则打印GPU加速提示
        if self.hardware_accelerator.has_accelerator():
            accelerator_name = self.hardware_accelerator.accelerator_name
            if accelerator_name == 'DirectML' and config.inpaintMode.value not in [InpaintMode.STTN_AUTO, InpaintMode.STTN_DET]:
                self.append_output(tr['Main']['DirectMLWarning'])
        os.makedirs(os.path.dirname(os.path.abspath(self.video_out_path)), exist_ok=True)
        # 重置进度条
        self.progress_total = 0
        tbar = tqdm(total=int(self.frame_count), unit='frame', position=0, file=sys.__stdout__,
                    desc='Subtitle Removing')
        if self.is_picture:
            original_frame = read_image(self.video_path)
            if original_frame is None:
                self.append_output(tr['Main']['ReadImageFailed'].format(self.video_path))
                return
            if config.inpaintMode.value in WATERMARK_INPAINT_MODES:
                mask = self._create_fixed_watermark_mask()
                _, outer_mask, _ = build_fixed_watermark_masks(mask)
                rgb_frame = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)
                completed_rgb = self.lama_inpaint.inpaint(rgb_frame, outer_mask)
                completed_frame = cv2.cvtColor(completed_rgb, cv2.COLOR_RGB2BGR)
                inpainted_frame, _ = self._blend_fixed_watermark_frame(original_frame, completed_frame, mask)
                self.push_preview_with_comp(
                    self._fixed_watermark_preview(original_frame, mask),
                    inpainted_frame,
                    force=True,
                )
            else:
                sub_detector = SubtitleDetect(self.video_path, self.sub_areas)
                sub_list = sub_detector.detect_subtitle(original_frame)
                del sub_detector
                gc.collect()
                if len(sub_list):
                    mask = create_mask(original_frame.shape[0:2], sub_list)
                    inpainted_frame = self.lama_inpaint.inpaint(original_frame, mask)
                    self.push_preview_with_comp(np.clip(original_frame+mask[:,:,np.newaxis]*0.3,0,255).astype(np.uint8), inpainted_frame, force=True)
                else:
                    inpainted_frame = original_frame
                    self.push_preview_with_comp(original_frame, inpainted_frame, force=True)
            cv2.imencode(self.ext, inpainted_frame)[1].tofile(self.video_out_path)
            tbar.update(1)
            self.progress_total = 100
        else:
            # 精准模式下，获取场景分割的帧号，进一步切割
            self.log_model()
            if config.inpaintMode.value == InpaintMode.PROPAINTER:
                self.propainter_mode(tbar)
            elif config.inpaintMode.value == InpaintMode.FIXED_WATERMARK:
                self.fixed_watermark_mode(tbar)
            elif config.inpaintMode.value == InpaintMode.MOVING_WATERMARK:
                self.moving_watermark_mode(tbar)
            elif config.inpaintMode.value == InpaintMode.STTN_AUTO:
                self.sttn_auto_mode(tbar)
            elif config.inpaintMode.value == InpaintMode.STTN_DET:
                self.video_inpaint(tbar, self.sttn_det_inpaint)
            elif config.inpaintMode.value == InpaintMode.LAMA:
                self.video_inpaint(tbar, self.lama_inpaint)
            elif config.inpaintMode.value == InpaintMode.OPENCV:
                self.video_inpaint(tbar, OpenCVInpaint())
            else:
                raise Exception(f'inpaint mode: {config.inpaintMode.value} not implemented')

        self.video_cap.release()
        self.video_writer.release()
        if not self.is_picture:
            # 将原音频合并到新生成的视频文件中
            self.merge_audio_to_video()
        self.append_output(tr['Main']['FinishedProcessing'].format(self.video_out_path))
        self.append_output(tr['Main']['ProcessingTime'].format(round(time.time() - start_time)))
        self.isFinished = True
        self.progress_total = 100
        if os.path.exists(self.video_temp_file.name):
            try:
                os.remove(self.video_temp_file.name)
            except Exception:
                pass #ignore

    def log_model(self):
        model_friendly_name = list(tr['InpaintMode'].values())[list(InpaintMode).index(config.inpaintMode.value)]
        model_device = 'CPU'
        if config.inpaintMode.value != InpaintMode.OPENCV and self.hardware_accelerator.has_accelerator():
            accelerator_name = self.hardware_accelerator.accelerator_name
            if accelerator_name == 'DirectML' and config.inpaintMode.value in [InpaintMode.STTN_AUTO, InpaintMode.STTN_DET]:
                model_device = 'DirectML'
            if self.hardware_accelerator.has_cuda() or self.hardware_accelerator.has_mps():
                model_device = accelerator_name
        self.append_output(tr['Main']['SubtitleRemoverModel'].format(f"{model_friendly_name} ({model_device})"))
        providers = ", ".join(self.hardware_accelerator.onnx_providers)
        providers_str = f" ({providers})" if providers else ""
        if config.inpaintMode.value not in WATERMARK_INPAINT_MODES:
            detect_mode_name = list(tr['SubtitleDetectMode'].values())[list(SubtitleDetectMode).index(config.subtitleDetectMode.value)]
            self.append_output(tr['Main']['SubtitleDetectionModel'].format(f"{detect_mode_name}{providers_str}"))

    def merge_audio_to_video(self):
        """Mux the original audio without an intermediate AAC extraction."""
        try:
            if not os.path.exists(self.video_temp_file.name):
                raise FileNotFoundError(self.video_temp_file.name)
            audio_merge_command = [
                FFmpegCLI.instance().ffmpeg_path,
                "-y",
                "-i", self.video_temp_file.name,
                "-i", self.video_path,
                "-map", "0:v:0",
                "-map", _preferred_audio_stream_spec(self.video_path),
                "-c:v", "copy",
                "-c:a", "copy",
                "-loglevel", "error",
                self.video_out_path,
            ]
            subprocess.check_output(
                audio_merge_command,
                stdin=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                timeout=600,
            )
        except Exception as e:
            traceback.print_exc()
            self.append_output(tr['Main']['FailToMergeAudio'].format(str(e)))
        else:
            self.is_successful_merged = True
        finally:
            if not self.is_successful_merged:
                try:
                    shutil.copy2(self.video_temp_file.name, self.video_out_path)
                except IOError as e:
                    self.append_output(tr['Main']['CopyFileFailed'].format(self.video_temp_file.name, self.video_out_path, str(e)))
            self.video_temp_file.close()

    @cached_property
    def lama_inpaint(self):
        model_path = os.path.join(self.model_config.LAMA_MODEL_DIR, 'big-lama.pt')
        device = self.hardware_accelerator.device if self.hardware_accelerator.has_cuda() or self.hardware_accelerator.has_mps() else torch.device("cpu")
        cache_key = (_device_cache_key(device), model_path)
        model = _LAMA_INPAINT_CACHE.get(cache_key)
        if model is None:
            model = LamaInpaint(device, model_path)
            _LAMA_INPAINT_CACHE[cache_key] = model
        return model

    @cached_property
    def sttn_det_inpaint(self):
        device = self.hardware_accelerator.device
        cache_key = (
            _device_cache_key(device),
            self.model_config.STTN_DET_MODEL_PATH,
            config.sttnNeighborStride.value,
            config.sttnReferenceLength.value,
        )
        model = _STTN_DET_INPAINT_CACHE.get(cache_key)
        if model is None:
            model = STTNDetInpaint(device, self.model_config.STTN_DET_MODEL_PATH)
            _STTN_DET_INPAINT_CACHE[cache_key] = model
        return model

    @cached_property
    def propainter_inpaint_model(self):
        device = self.hardware_accelerator.device if self.hardware_accelerator.has_cuda() else torch.device("cpu")
        max_load_num = max(1, int(self.propainter_max_load_num))
        cache_key = (
            _device_cache_key(device),
            self.model_config.PROPAINTER_MODEL_DIR,
        )
        model = _PROPAINTER_INPAINT_CACHE.get(cache_key)
        if model is None:
            model = PropainterInpaint(
                device,
                self.model_config.PROPAINTER_MODEL_DIR,
                max_load_num,
            )
            _PROPAINTER_INPAINT_CACHE[cache_key] = model
        else:
            # Batch length changes inference chunking only; reuse the same
            # weights instead of retaining a full model copy per setting.
            model.sub_video_length = max_load_num
        return model


if __name__ == '__main__':
    multiprocessing.set_start_method("spawn")
    from backend.tools.args_handler import parse_args
    args = parse_args()
    # force english
    config.set(config.interface, 'en')
    TRANSLATION_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'interface', f"{config.interface.value}.ini")
    tr.read(TRANSLATION_FILE, encoding='utf-8')
    sr = SubtitleRemover(args.input)
    if not is_video_or_image(args.input):
        sr.append_output(f'Error: {args.input} is not supported or is corrupted.')
        exit(-1)
    sr.sub_areas = args.subtitle_area_coords
    if args.output is not None:
        sr.video_out_path = args.output
    config.inpaintMode.value = args.inpaint_mode
    sr.run()
        
