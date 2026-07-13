import os
import cv2
import math
import re
import threading
import multiprocessing
import tempfile
import time
import traceback
import unicodedata
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeoutError
from collections import Counter
from PySide6.QtWidgets import QWidget, QHBoxLayout, QVBoxLayout
from PySide6.QtCore import Slot, QRect, Signal
from PySide6 import QtWidgets
from datetime import datetime
from qfluentwidgets import (PushButton, CardWidget, TextEdit, FluentIcon, qconfig)
from ui.setting_interface import SettingInterface
from ui.component.video_display_component import VideoDisplayComponent
from ui.component.task_list_component import TaskListComponent, TaskStatus, TaskOptions
from ui.icon.my_fluent_icon import MyFluentIcon
from backend.config import config, tr
from backend.tools.constant import InpaintMode
from backend.tools.subtitle_detect import auto_detect_subtitle_area, detect_subtitle_intervals
from backend.tools.subtitle_remover_remote_call import SubtitleRemoverRemoteCall
from backend.tools.process_manager import ProcessManager
from backend.tools.common_tools import get_readable_path, is_image_file, is_video_file, read_image
from backend.tools.watermark_tracker import (
    build_moving_watermark_preprocess_key,
    preprocess_moving_watermark_to_file,
)

WATERMARK_INPAINT_MODES = frozenset((
    InpaintMode.FIXED_WATERMARK,
    InpaintMode.MOVING_WATERMARK,
))

_CHINESE_DIGITS = {
    "零": 0,
    "〇": 0,
    "一": 1,
    "二": 2,
    "两": 2,
    "三": 3,
    "四": 4,
    "五": 5,
    "六": 6,
    "七": 7,
    "八": 8,
    "九": 9,
}
_CHINESE_SMALL_UNITS = {"十": 10, "百": 100, "千": 1000}
_CHINESE_LARGE_UNITS = {"万": 10_000, "亿": 100_000_000}
_CHINESE_EPISODE_PATTERN = re.compile(
    r"第\s*([零〇一二两三四五六七八九十百千万亿]+)\s*([集话話回期章])"
)
_NATURAL_NUMBER_PATTERN = re.compile(r"(\d+)")


def _chinese_number_to_int(text):
    """Convert common Chinese episode numerals to an integer."""
    if not text:
        return None
    if not any(char in _CHINESE_SMALL_UNITS or char in _CHINESE_LARGE_UNITS for char in text):
        try:
            return int("".join(str(_CHINESE_DIGITS[char]) for char in text))
        except (KeyError, ValueError):
            return None

    total = 0
    section = 0
    number = 0
    for char in text:
        if char in _CHINESE_DIGITS:
            number = _CHINESE_DIGITS[char]
            continue
        if char in _CHINESE_SMALL_UNITS:
            section += (number or 1) * _CHINESE_SMALL_UNITS[char]
            number = 0
            continue
        if char in _CHINESE_LARGE_UNITS:
            section = (section + number) * _CHINESE_LARGE_UNITS[char]
            total += section
            section = 0
            number = 0
            continue
        return None
    return total + section + number


def _episode_name_sort_key(path):
    """Return a deterministic natural key for Arabic and Chinese episode names."""
    basename = os.path.basename(os.fspath(path))
    normalized = unicodedata.normalize("NFKC", basename).casefold()

    def replace_chinese_episode(match):
        episode_number = _chinese_number_to_int(match.group(1))
        if episode_number is None:
            return match.group(0)
        return f"第{episode_number}{match.group(2)}"

    normalized = _CHINESE_EPISODE_PATTERN.sub(replace_chinese_episode, normalized)
    natural_parts = tuple(
        (0, int(part), len(part)) if part.isdigit() else (1, part, 0)
        for part in _NATURAL_NUMBER_PATTERN.split(normalized)
        if part
    )
    return natural_parts, normalized, basename


class BatchFolderManagerDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, default_output_root=""):
        super().__init__(parent)
        self.setWindowTitle("\u6279\u91cf\u6587\u4ef6\u5939\u7ba1\u7406")
        self.resize(720, 480)
        self.selected_folders = []
        self.output_root = default_output_root

        layout = QVBoxLayout(self)

        folder_label = QtWidgets.QLabel("\u5df2\u9009\u6587\u4ef6\u5939")
        layout.addWidget(folder_label)

        self.folder_list = QtWidgets.QListWidget(self)
        self.folder_list.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        layout.addWidget(self.folder_list, 1)

        folder_button_layout = QHBoxLayout()
        self.add_folder_button = PushButton("\u6dfb\u52a0\u6587\u4ef6\u5939", self)
        self.add_folder_button.clicked.connect(self.add_folder)
        folder_button_layout.addWidget(self.add_folder_button)

        self.remove_folder_button = PushButton("\u5220\u9664\u9009\u4e2d", self)
        self.remove_folder_button.clicked.connect(self.remove_selected_folders)
        folder_button_layout.addWidget(self.remove_folder_button)

        self.clear_folder_button = PushButton("\u6e05\u7a7a", self)
        self.clear_folder_button.clicked.connect(self.clear_folders)
        folder_button_layout.addWidget(self.clear_folder_button)
        folder_button_layout.addStretch(1)
        layout.addLayout(folder_button_layout)

        output_label = QtWidgets.QLabel("\u8f93\u51fa\u6839\u76ee\u5f55")
        layout.addWidget(output_label)

        output_layout = QHBoxLayout()
        self.output_root_edit = QtWidgets.QLineEdit(self)
        self.output_root_edit.setReadOnly(True)
        self.output_root_edit.setText(self.output_root)
        output_layout.addWidget(self.output_root_edit, 1)

        self.choose_output_button = PushButton("\u9009\u62e9\u8f93\u51fa\u6839\u76ee\u5f55", self)
        self.choose_output_button.clicked.connect(self.choose_output_root)
        output_layout.addWidget(self.choose_output_button)
        layout.addLayout(output_layout)

        bottom_layout = QHBoxLayout()
        bottom_layout.addStretch(1)
        self.cancel_button = PushButton("\u53d6\u6d88", self)
        self.cancel_button.clicked.connect(self.reject)
        bottom_layout.addWidget(self.cancel_button)
        self.confirm_button = PushButton("\u5f00\u59cb\u5bfc\u5165", self)
        self.confirm_button.clicked.connect(self.accept_if_valid)
        bottom_layout.addWidget(self.confirm_button)
        layout.addLayout(bottom_layout)

    def add_folder(self):
        folders = self.select_multiple_directories(
            "\u9009\u62e9\u6587\u4ef6\u5939",
            self.selected_folders[-1] if self.selected_folders else ""
        )
        if not folders:
            return
        for folder in folders:
            folder = os.path.normpath(folder)
            if folder in self.selected_folders:
                continue
            self.selected_folders.append(folder)
            self.folder_list.addItem(folder)

    def select_multiple_directories(self, title, initial_dir=""):
        dialog = QtWidgets.QFileDialog(self, title, initial_dir)
        dialog.setFileMode(QtWidgets.QFileDialog.Directory)
        dialog.setOption(QtWidgets.QFileDialog.DontUseNativeDialog, True)
        dialog.setOption(QtWidgets.QFileDialog.ShowDirsOnly, True)
        for view in dialog.findChildren(QtWidgets.QListView):
            view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        for view in dialog.findChildren(QtWidgets.QTreeView):
            view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return []
        folders = []
        for folder in dialog.selectedFiles():
            folder = os.path.normpath(folder)
            if folder not in folders:
                folders.append(folder)
        return folders

    def remove_selected_folders(self):
        selected_items = self.folder_list.selectedItems()
        if not selected_items:
            return
        selected_paths = {item.text() for item in selected_items}
        self.selected_folders = [folder for folder in self.selected_folders if folder not in selected_paths]
        for item in selected_items:
            self.folder_list.takeItem(self.folder_list.row(item))

    def clear_folders(self):
        self.selected_folders = []
        self.folder_list.clear()

    def choose_output_root(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "\u9009\u62e9\u8f93\u51fa\u6839\u76ee\u5f55",
            self.output_root or (self.selected_folders[0] if self.selected_folders else "")
        )
        if not folder:
            return
        self.output_root = os.path.normpath(folder)
        self.output_root_edit.setText(self.output_root)

    def accept_if_valid(self):
        if not self.selected_folders:
            QtWidgets.QMessageBox.warning(self, "\u63d0\u793a", "\u8bf7\u5148\u6dfb\u52a0\u81f3\u5c11\u4e00\u4e2a\u6587\u4ef6\u5939")
            return
        if not self.output_root:
            QtWidgets.QMessageBox.warning(self, "\u63d0\u793a", "\u8bf7\u9009\u62e9\u8f93\u51fa\u6839\u76ee\u5f55")
            return
        self.accept()

class HomeInterface(QWidget):
    progress_signal = Signal(int, bool)
    append_log_signal = Signal(list)
    update_preview_with_comp_signal = Signal(list)
    task_error_signal = Signal(object)
    toggle_buttons_signal = Signal(bool)  # True=显示运行按钮, False=显示停止按钮
    task_status_signal = Signal(int, object)  # (task_index, TaskStatus)
    select_task_signal = Signal(int)  # task_index
    auto_subtitle_area_signal = Signal(list, float)
    auto_subtitle_area_error_signal = Signal(str)
    auto_subtitle_area_running_signal = Signal(bool)
    processing_phase_signal = Signal(str, str)
    def __init__(self, parent=None):
        super().__init__(parent=parent)
        self.setObjectName("HomeInterface")
        # 初始化一些变量
        self.video_path = None
        self.video_cap = None
        self.fps = None
        self.frame_count = None
        self.frame_width = None
        self.frame_height = None
        self._displayed_frame_no = 0
        self.se = None  # 后台字幕提取器

        # 字幕区域参数
        self.xmin = None
        self.xmax = None
        self.ymin = None
        self.ymax = None

        # 添加自动滚动控制标志
        self.auto_scroll = True
        self._stop_event = threading.Event()  # 线程安全的停止信号
        self._worker_thread = None
        self.running_process = None
        self._saved_inpaint_mode = None  # 保存图片锁定前的 inpaint 模式
        self._video_cap_lock = threading.Lock()  # 保护 video_cap 的线程锁
        self._selection_change_source = None
        self._auto_area_button_running = False
        self._is_processing = False
        self.current_processing_batch_id = None
        self.current_processing_task_start_time = None
        self.current_processing_run_start_time = None
        self.auto_area_executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="auto-area")
        self._auto_area_lock = threading.Lock()
        self._auto_area_futures = {}
        self._auto_area_results = {}
        self._auto_area_errors = {}
        self._subtitle_interval_futures = {}
        self._subtitle_interval_results = {}
        self._subtitle_interval_errors = {}
        self._video_dimension_cache = {}
        self._video_frame_count_cache = {}
        self._video_fps_cache = {}
        self.moving_preprocess_executor = ThreadPoolExecutor(
            max_workers=1,
            thread_name_prefix="moving-watermark-preprocess",
        )
        self._moving_preprocess_lock = threading.Lock()
        self._moving_preprocess_futures = {}
        self._moving_preprocess_results = {}
        self._moving_preprocess_errors = {}
        self._moving_preprocess_artifacts = set()
        self._moving_preprocess_generation = 0
        self._moving_preprocess_cancel_event = threading.Event()
        self._active_run_mode = None

        # 当前正在处理的任务索引
        self.current_processing_task_index = -1
        self.worker_process = None
        self.worker_command_queue = None
        self.worker_remote_caller = None
        self.last_worker_job_succeeded = False

        self.__init_widgets()
        self.progress_signal.connect(self.update_progress)
        self.append_log_signal.connect(self.append_log)
        self.update_preview_with_comp_signal.connect(self.update_preview_with_comp)
        self.task_error_signal.connect(self.on_task_error)
        self.toggle_buttons_signal.connect(self._toggle_buttons)
        self.task_status_signal.connect(lambda idx, status: self.task_list_component.update_task_status(idx, status))
        self.select_task_signal.connect(self.task_list_component.select_task)
        self.auto_subtitle_area_signal.connect(self.on_auto_subtitle_area_detected)
        self.auto_subtitle_area_error_signal.connect(lambda message: self.append_output(message))
        self.auto_subtitle_area_running_signal.connect(self.set_auto_area_button_running)
        self.processing_phase_signal.connect(self.on_processing_phase)
        config.inpaintMode.valueChanged.connect(self.on_inpaint_mode_changed)
        self.on_inpaint_mode_changed(config.inpaintMode.value)

    def __init_widgets(self):
        """创建主页面"""
        main_layout = QHBoxLayout(self)
        main_layout.setSpacing(8)
        main_layout.setContentsMargins(16, 16, 16, 16)

        # 左侧视频区域
        left_layout = QVBoxLayout()
        left_layout.setSpacing(8)
        
        # 创建视频显示组件
        self.video_display_component = VideoDisplayComponent(self)
        self.video_display_component.ab_sections_changed.connect(self.ab_sections_changed)
        self.video_display_component.selections_changed.connect(self.selections_changed)
        left_layout.addWidget(self.video_display_component)
        
        # 获取视频显示和滑块的引用
        self.video_display = self.video_display_component.video_display
        self.video_slider = self.video_display_component.video_slider
        self.video_slider.valueChanged.connect(self.slider_changed)
        
        # 输出文本区域
        self.output_text = TextEdit()
        self.output_text.setMinimumHeight(150)
        self.output_text.setReadOnly(True)
        self.output_text.document().setDocumentMargin(10)        
        # 连接滚动条值变化信号
        self.output_text.verticalScrollBar().valueChanged.connect(self.on_scroll_change)
        
        output_container = CardWidget(self)
        output_layout = QVBoxLayout()
        output_layout.setContentsMargins(0, 0, 0, 0)
        output_layout.addWidget(self.output_text)
        output_container.setLayout(output_layout)
        left_layout.addWidget(output_container)

        main_layout.addLayout(left_layout, 2)

        # 右侧设置区域
        right_layout = QVBoxLayout()
        right_layout.setSpacing(10)

        # 设置容器
        settings_container = CardWidget(self)
        self.setting_interface = SettingInterface(settings_container)
        settings_container.setLayout(self.setting_interface)
        right_layout.addWidget(settings_container)
        
        # 添加任务列表容器
        task_list_container = CardWidget(self)
        task_list_layout = QHBoxLayout()
        task_list_layout.setContentsMargins(0, 0, 0, 0)
        task_list_layout.setSpacing(0)
        self.task_list_component = TaskListComponent(self)
        self.task_list_component.task_selected.connect(self.on_task_selected)
        self.task_list_component.task_deleted.connect(self.on_task_deleted)
        task_list_layout.addWidget(self.task_list_component)
        task_list_container.setLayout(task_list_layout)
        right_layout.addWidget(task_list_container, 1)  # 占满剩余空间
        
        # 操作按钮容器
        button_container = CardWidget(self)
        button_layout = QHBoxLayout()
        button_layout.setContentsMargins(16, 16, 16, 16)
        button_layout.setSpacing(8)
        
        self.file_button = PushButton(tr['SubtitleExtractorGUI']['Open'], self)
        self.file_button.setIcon(FluentIcon.FOLDER)
        self.file_button.clicked.connect(self.open_file)
        button_layout.addWidget(self.file_button)

        self.folder_button = PushButton("\u6253\u5f00\u6587\u4ef6\u5939", self)
        self.folder_button.setIcon(FluentIcon.FOLDER)
        self.folder_button.clicked.connect(self.open_folder)
        button_layout.addWidget(self.folder_button)

        self.batch_folder_button = PushButton("\u6279\u91cf\u6587\u4ef6\u5939", self)
        self.batch_folder_button.setIcon(FluentIcon.FOLDER)
        self.batch_folder_button.clicked.connect(self.open_folders_batch)
        button_layout.addWidget(self.batch_folder_button)

        self.auto_area_button = PushButton("自动框选", self)
        self.auto_area_button.setIcon(FluentIcon.SEARCH)
        self.auto_area_button.clicked.connect(self.auto_area_button_clicked)
        button_layout.addWidget(self.auto_area_button)
        
        self.run_button = PushButton(tr['SubtitleExtractorGUI']['Run'], self)
        self.run_button.setIcon(FluentIcon.PLAY)
        self.run_button.clicked.connect(self.run_button_clicked)
        button_layout.addWidget(self.run_button)
        
        self.stop_button = PushButton(tr['SubtitleExtractorGUI']['Stop'], self)
        self.stop_button.setIcon(MyFluentIcon.Stop)
        self.stop_button.setVisible(False)
        self.stop_button.clicked.connect(self.stop_button_clicked)
        
        button_layout.addWidget(self.stop_button)
        
        button_container.setLayout(button_layout)
        right_layout.addWidget(button_container)

        main_layout.addLayout(right_layout, 1)
    
    def on_scroll_change(self, value):
        """监控滚动条位置变化"""
        scrollbar = self.output_text.verticalScrollBar()
        # 如果滚动到底部，启用自动滚动
        if value == scrollbar.maximum():
            self.auto_scroll = True
        # 如果用户向上滚动，禁用自动滚动
        elif self.auto_scroll and value < scrollbar.maximum():
            self.auto_scroll = False

    
    def slider_changed(self, value):
        frame = None
        with self._video_cap_lock:
            if self.video_cap is not None and self.video_cap.isOpened():
                frame_no = int(value)
                self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, frame_no)
                ret, frame = self.video_cap.read()
                if not ret:
                    frame = None
                else:
                    self._displayed_frame_no = max(
                        0,
                        int(round(self.video_cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1,
                    )
        if frame is not None:
            # 更新预览图像
            self.update_preview(frame)

    def _seek_preview_to_frame(self, frame_no):
        """Decode and display an exact zero-based reference frame."""
        try:
            frame_no = int(frame_no)
        except (TypeError, ValueError):
            return False
        if frame_no < 0 or not self.frame_count or frame_no >= self.frame_count:
            return False
        self.video_slider.blockSignals(True)
        try:
            self.video_slider.setValue(
                max(self.video_slider.minimum(), min(frame_no, self.video_slider.maximum()))
            )
        finally:
            self.video_slider.blockSignals(False)
        self.slider_changed(frame_no)
        return self._displayed_frame_no == frame_no

    @staticmethod
    def _is_watermark_mode(mode=None):
        return (mode or config.inpaintMode.value) in WATERMARK_INPAINT_MODES

    def ab_sections_changed(self, ab_sections):
        if self._is_processing:
            return
        get_current_task_index = self.task_list_component.get_current_task_index()
        if get_current_task_index == -1:
            return
        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.AB_SECTIONS, ab_sections)

    def selections_changed(self, selections):
        if self._is_processing:
            return
        get_current_task_index = self.task_list_component.get_current_task_index()
        if get_current_task_index == -1:
            return
        if config.inpaintMode.value == InpaintMode.FIXED_WATERMARK:
            normalized_areas = self._sanitize_normalized_areas(
                self.video_display_component.preview_coordinates_to_normalized_video_coordinates(
                    selections
                )
            )
            self.task_list_component.update_task_option(
                get_current_task_index,
                TaskOptions.FIXED_WATERMARK_AREAS,
                normalized_areas,
            )
            if self._selection_change_source is None:
                self._save_fixed_watermark_areas(normalized_areas)
                self._propagate_fixed_watermark_areas(get_current_task_index, normalized_areas)
            return

        if config.inpaintMode.value == InpaintMode.MOVING_WATERMARK:
            # The tracking MVP intentionally uses one tight template. Keep the
            # most recently created/active selection if the generic selector
            # contains more than one rectangle.
            selections = list(selections[-1:]) if selections else []
            if selections != self.video_display_component.get_selection_rects():
                self.video_display_component.set_selection_rects(selections)
            # Loading an existing task/template also emits selectionsChanged.
            # It is display-only: keep the template source and reference frame
            # captured from the original batch video instead of overwriting
            # them with the task that has just been selected.
            if self._selection_change_source is not None:
                return
            normalized_areas = self._sanitize_normalized_areas(
                self.video_display_component.preview_coordinates_to_normalized_video_coordinates(
                    selections
                )
            )
            normalized_area = normalized_areas[-1] if normalized_areas else None
            reference_frame_no = self._displayed_frame_no if normalized_area else None
            template_source_path = self.video_path if normalized_area else None
            self.task_list_component.update_task_option(
                get_current_task_index,
                TaskOptions.MOVING_WATERMARK_TEMPLATE_AREA,
                normalized_area,
            )
            self.task_list_component.update_task_option(
                get_current_task_index,
                TaskOptions.MOVING_WATERMARK_REFERENCE_FRAME_NO,
                reference_frame_no,
            )
            self.task_list_component.update_task_option(
                get_current_task_index,
                TaskOptions.MOVING_WATERMARK_TEMPLATE_SOURCE_PATH,
                template_source_path,
            )
            if self._selection_change_source is None:
                self._propagate_moving_watermark_template(
                    get_current_task_index,
                    normalized_area,
                    reference_frame_no,
                    template_source_path,
                )
            return

        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.SUB_AREAS, selections)
        source = self._selection_change_source or "manual"
        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.SUB_AREAS_SOURCE, source)
        self.task_list_component.update_task_option(get_current_task_index, TaskOptions.SUBTITLE_INTERVALS, None)

    @staticmethod
    def _parse_fixed_watermark_areas(value):
        areas = []
        for area in (value or "").split(";"):
            if not area:
                continue
            try:
                ymin, ymax, xmin, xmax = map(float, area.split(","))
            except (TypeError, ValueError):
                continue
            areas.append((ymin, ymax, xmin, xmax))
        return HomeInterface._sanitize_normalized_areas(areas)

    @staticmethod
    def _sanitize_normalized_areas(areas):
        sanitized = []
        for area in areas or []:
            try:
                ymin, ymax, xmin, xmax = map(float, area)
            except (TypeError, ValueError):
                continue
            if not all(math.isfinite(value) for value in (ymin, ymax, xmin, xmax)):
                continue
            ymin, ymax = sorted((max(0.0, min(1.0, ymin)), max(0.0, min(1.0, ymax))))
            xmin, xmax = sorted((max(0.0, min(1.0, xmin)), max(0.0, min(1.0, xmax))))
            if ymax <= ymin or xmax <= xmin:
                continue
            sanitized.append((ymin, ymax, xmin, xmax))
        return sanitized

    @staticmethod
    def _save_fixed_watermark_areas(areas):
        areas = HomeInterface._sanitize_normalized_areas(areas)
        config.fixedWatermarkSelectionAreas.value = ";".join(
            f"{ymin:.6f},{ymax:.6f},{xmin:.6f},{xmax:.6f}"
            for ymin, ymax, xmin, xmax in areas
        )
        qconfig.save()

    def _get_video_dimensions(self, path):
        dimensions = self._video_dimension_cache.get(path)
        if dimensions:
            return dimensions
        cap = cv2.VideoCapture(get_readable_path(path))
        dimensions = (
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
        )
        cap.release()
        if dimensions[0] > 0 and dimensions[1] > 0:
            self._video_dimension_cache[path] = dimensions
            return dimensions
        return None

    def _get_video_frame_count(self, path):
        frame_count = self._video_frame_count_cache.get(path)
        if frame_count is not None:
            return frame_count
        cap = cv2.VideoCapture(get_readable_path(path))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if cap.isOpened() else 0
        cap.release()
        if frame_count > 0:
            self._video_frame_count_cache[path] = frame_count
        return frame_count

    def _get_video_fps(self, path):
        fps = self._video_fps_cache.get(path)
        if fps is not None:
            return fps
        cap = cv2.VideoCapture(get_readable_path(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) if cap.isOpened() else 0.0
        cap.release()
        if fps > 0:
            self._video_fps_cache[path] = fps
        return fps

    @staticmethod
    def _normalized_areas_to_video_coordinates(normalized_areas, dimensions):
        if not dimensions:
            return []
        width, height = dimensions
        video_areas = []
        for ymin, ymax, xmin, xmax in HomeInterface._sanitize_normalized_areas(normalized_areas):
            video_area = (
                round(ymin * height),
                round(ymax * height),
                round(xmin * width),
                round(xmax * width),
            )
            if video_area[1] > video_area[0] and video_area[3] > video_area[2]:
                video_areas.append(video_area)
        return video_areas

    def _propagate_fixed_watermark_areas(self, task_index, normalized_areas):
        """Apply a manual watermark selection to pending videos in the same aspect-ratio batch."""
        normalized_areas = self._sanitize_normalized_areas(normalized_areas)
        task = self.task_list_component.get_task(task_index)
        if task is None:
            return
        target_dimensions = self._get_video_dimensions(task.path)
        target_ratio = (
            target_dimensions[0] / target_dimensions[1]
            if target_dimensions and target_dimensions[1]
            else None
        )
        for pending_index, pending_task in self.task_list_component.get_pending_tasks_by_batch(task.batch_id):
            if target_ratio and pending_task.path != task.path:
                dimensions = self._get_video_dimensions(pending_task.path)
                if not dimensions or not dimensions[1]:
                    continue
                ratio = dimensions[0] / dimensions[1]
                if abs(ratio - target_ratio) / target_ratio > 0.01:
                    continue
            self.task_list_component.update_task_option(
                pending_index,
                TaskOptions.FIXED_WATERMARK_AREAS,
                list(normalized_areas),
            )

    def _propagate_moving_watermark_template(
        self,
        task_index,
        normalized_area,
        reference_frame_no,
        template_source_path,
    ):
        """Share one captured tracking template with pending tasks in the batch."""
        task = self.task_list_component.get_task(task_index)
        if task is None:
            return
        normalized = self._sanitize_normalized_areas(
            [normalized_area] if normalized_area is not None else []
        )
        normalized_area = normalized[0] if len(normalized) == 1 else None
        if normalized_area is None:
            reference_frame_no = None
            template_source_path = None
        target_dimensions = self._get_video_dimensions(task.path)
        target_ratio = (
            target_dimensions[0] / target_dimensions[1]
            if target_dimensions and target_dimensions[1]
            else None
        )
        for pending_index, pending_task in self.task_list_component.get_pending_tasks_by_batch(task.batch_id):
            if normalized_area is not None and target_ratio and pending_task.path != task.path:
                dimensions = self._get_video_dimensions(pending_task.path)
                if not dimensions or not dimensions[1]:
                    continue
                ratio = dimensions[0] / dimensions[1]
                if abs(ratio - target_ratio) / target_ratio > 0.01:
                    continue
            self.task_list_component.update_task_option(
                pending_index,
                TaskOptions.MOVING_WATERMARK_TEMPLATE_AREA,
                normalized_area,
            )
            self.task_list_component.update_task_option(
                pending_index,
                TaskOptions.MOVING_WATERMARK_REFERENCE_FRAME_NO,
                reference_frame_no,
            )
            self.task_list_component.update_task_option(
                pending_index,
                TaskOptions.MOVING_WATERMARK_TEMPLATE_SOURCE_PATH,
                template_source_path,
            )

    def _moving_watermark_template_for_task(self, task_index):
        area = self.task_list_component.get_task_option(
            task_index,
            TaskOptions.MOVING_WATERMARK_TEMPLATE_AREA,
            None,
        )
        normalized = self._sanitize_normalized_areas([area] if area is not None else [])
        reference_frame_no = self.task_list_component.get_task_option(
            task_index,
            TaskOptions.MOVING_WATERMARK_REFERENCE_FRAME_NO,
            None,
        )
        if isinstance(reference_frame_no, bool):
            reference_frame_no = None
        else:
            try:
                numeric_frame_no = float(reference_frame_no)
                parsed_frame_no = int(numeric_frame_no)
                reference_frame_no = (
                    parsed_frame_no if numeric_frame_no == parsed_frame_no else None
                )
            except (TypeError, ValueError, OverflowError):
                reference_frame_no = None
        template_source_path = self.task_list_component.get_task_option(
            task_index,
            TaskOptions.MOVING_WATERMARK_TEMPLATE_SOURCE_PATH,
            None,
        )
        try:
            template_source_path = os.fspath(template_source_path) if template_source_path else None
        except TypeError:
            template_source_path = None
        return (
            normalized[0] if len(normalized) == 1 else None,
            reference_frame_no,
            template_source_path,
        )

    def on_task_selected(self, index, file_path):
        """处理任务被选中事件
        
        Args:
            index: 任务索引
            file_path: 文件路径
        """
        # 加载选中的视频进行预览
        self.load_video(file_path)
        ab_sections = self.task_list_component.get_task_option(index, TaskOptions.AB_SECTIONS, [])
        self.video_display_component.set_ab_sections(ab_sections)
        selections = self.task_list_component.get_task_option(index, TaskOptions.SUB_AREAS, [])
        if config.inpaintMode.value == InpaintMode.FIXED_WATERMARK:
            normalized_areas = self._sanitize_normalized_areas(
                self.task_list_component.get_task_option(
                    index, TaskOptions.FIXED_WATERMARK_AREAS, []
                )
            )
            loaded_from_config = False
            if not normalized_areas:
                normalized_areas = self._parse_fixed_watermark_areas(
                    config.fixedWatermarkSelectionAreas.value
                )
                if normalized_areas:
                    loaded_from_config = True
                    self.task_list_component.update_task_option(
                        index,
                        TaskOptions.FIXED_WATERMARK_AREAS,
                        normalized_areas,
                    )
            if loaded_from_config:
                self._propagate_fixed_watermark_areas(index, normalized_areas)
            self._selection_change_source = "fixed_config"
            try:
                self.video_display_component.set_selection_rects(
                    self.video_display_component.normalized_video_coordinates_to_preview_coordinates(
                        normalized_areas
                    )
                )
            finally:
                self._selection_change_source = None
            return
        if config.inpaintMode.value == InpaintMode.MOVING_WATERMARK:
            normalized_area, reference_frame_no, template_source_path = (
                self._moving_watermark_template_for_task(index)
            )
            if (
                normalized_area is not None
                and template_source_path
                and os.path.abspath(template_source_path) == os.path.abspath(file_path)
                and reference_frame_no is not None
            ):
                self._seek_preview_to_frame(reference_frame_no)
            self._selection_change_source = "moving_template"
            try:
                preview_areas = self.video_display_component.normalized_video_coordinates_to_preview_coordinates(
                    [normalized_area] if normalized_area is not None else []
                )
                self.video_display_component.set_selection_rects(preview_areas)
            finally:
                self._selection_change_source = None
            return
        if len(selections) <= 0:
            self._selection_change_source = "default"
            try:
                self.video_display_component.load_selections_from_config()
            finally:
                self._selection_change_source = None
        else:
            self.video_display_component.set_selection_rects(selections)
    
    def on_task_deleted(self, index):
        """处理任务被删除事件
        
        Args:
            index: 任务索引
        """
        # 如果删除的是正在处理的任务，则需要更新状态
        if index == self.current_processing_task_index:
            self.current_processing_task_index = -1
        
        task = self.task_list_component.get_task(0)
        if task:
            # 如果还有任务，选中第一个
            self.task_list_component.select_task(0)

    def update_preview(self, frame):
        # 先缩放图像
        resized_frame = self._img_resize(frame)

        # 设置视频参数
        self.video_display_component.set_video_parameters(
            self.frame_width, self.frame_height, 
            self.scaled_width if hasattr(self, 'scaled_width') else None,
            self.scaled_height if hasattr(self, 'scaled_height') else None,
            self.border_left if hasattr(self, 'border_left') else 0,
            self.border_top if hasattr(self, 'border_top') else 0,
            self.fps if self.fps is not None else 30,
        )
        
        # 更新视频显示（这会同时保存current_pixmap）
        self.video_display_component.update_video_display(resized_frame)

    def _img_resize(self, image):
        height, width = image.shape[:2]
        
        video_preview_width = self.video_display_component.video_preview_width
        video_preview_height = self.video_display_component.video_preview_height
        # 计算等比缩放后的尺寸
        target_ratio = video_preview_width / video_preview_height
        image_ratio = width / height
        
        if image_ratio > target_ratio:
            # 宽度适配，高度按比例缩放
            new_width = video_preview_width
            new_height = int(new_width / image_ratio)
            top_border = (video_preview_height - new_height) // 2
            bottom_border = video_preview_height - new_height - top_border
            left_border = 0
            right_border = 0
        else:
            # 高度适配，宽度按比例缩放
            new_height = video_preview_height
            new_width = int(new_height * image_ratio)
            left_border = (video_preview_width - new_width) // 2
            right_border = video_preview_width - new_width - left_border
            top_border = 0
            bottom_border = 0
        
        # 先缩放图像
        resized = cv2.resize(image, (new_width, new_height))
        
        # 添加黑边以填充到目标尺寸
        padded = cv2.copyMakeBorder(
            resized, 
            top_border, bottom_border, 
            left_border, right_border, 
            cv2.BORDER_CONSTANT, 
            value=[0, 0, 0]
        )
        
        # 保存边框信息，用于坐标转换
        self.border_left = left_border / video_preview_width
        self.border_right = right_border / video_preview_width
        self.border_top = top_border / video_preview_height
        self.border_bottom = bottom_border / video_preview_height
        self.original_width = width
        self.original_height = height
        self.is_vertical = width < height
        self.scaled_width = new_width / video_preview_width
        self.scaled_height = new_height / video_preview_height
        
        return padded

    def stop_button_clicked(self):
        try:
            self._stop_event.set()
            self._cancel_moving_preprocessing()
            running_process = self.running_process
            if running_process:
                self._dispose_subtitle_worker(terminate=True)
            # 更新任务状态为待处理
            if self.current_processing_task_index >= 0:
                self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.PENDING)
        finally:
            self._is_processing = False
            self.running_process = None
            self._toggle_buttons(True)
            self.current_processing_batch_id = None
            self.current_processing_task_start_time = None
            self.current_processing_run_start_time = None

    def _register_worker_callbacks(self, remote_caller):
        remote_caller.register_update_progress_callback(self.progress_signal.emit)
        remote_caller.register_log_callback(self.append_log_signal.emit)
        remote_caller.register_update_preview_with_comp_callback(self.update_preview_with_comp_signal.emit)
        remote_caller.register_error_callback(self.task_error_signal.emit)
        remote_caller.register_processing_phase_callback(self.processing_phase_signal.emit)

    def _ensure_subtitle_worker(self):
        if (
            self.worker_process
            and self.worker_process.is_alive()
            and self.worker_command_queue is not None
            and self.worker_remote_caller is not None
        ):
            self.running_process = self.worker_process
            return self.worker_process

        self._dispose_subtitle_worker(terminate=False)
        remote_caller = SubtitleRemoverRemoteCall()
        self._register_worker_callbacks(remote_caller)
        command_queue = multiprocessing.Queue()
        process = multiprocessing.Process(
            target=HomeInterface.remover_worker_process,
            args=(command_queue, remote_caller.queue)
        )
        process.start()
        ProcessManager.instance().add_process(process)
        self.worker_process = process
        self.worker_command_queue = command_queue
        self.worker_remote_caller = remote_caller
        self.running_process = process
        return process

    def _dispose_subtitle_worker(self, terminate=False):
        process = self.worker_process
        command_queue = self.worker_command_queue
        remote_caller = self.worker_remote_caller
        self.worker_process = None
        self.worker_command_queue = None
        self.worker_remote_caller = None

        if process:
            if terminate:
                ProcessManager.instance().terminate_by_process(process)
            else:
                if process.is_alive() and command_queue is not None:
                    try:
                        command_queue.put(("shutdown",))
                    except Exception:
                        pass
                if process.is_alive():
                    process.join(timeout=3)
                if process.is_alive():
                    ProcessManager.instance().terminate_by_process(process)

        if remote_caller:
            remote_caller.stop()

        self.running_process = None
        self.last_worker_job_succeeded = False

    @Slot(bool)
    def _toggle_buttons(self, show_run):
        """线程安全地切换按钮可见性"""
        self._is_processing = not show_run
        self.run_button.setVisible(show_run)
        self.stop_button.setVisible(not show_run)
        self.video_display_component.set_dragger_enabled(show_run)
        self.task_list_component.setEnabled(show_run)
        self.file_button.setEnabled(show_run)
        self.folder_button.setEnabled(show_run)
        self.batch_folder_button.setEnabled(show_run)
        self.auto_area_button.setEnabled(
            show_run
            and not self._auto_area_button_running
            and not self._is_watermark_mode()
        )
        self.setting_interface.set_inpaint_mode_enabled(
            show_run and self._saved_inpaint_mode is None
        )

    @Slot(bool)
    def set_auto_area_button_running(self, running):
        self._auto_area_button_running = running
        watermark_mode = self._is_watermark_mode()
        self.auto_area_button.setEnabled(not running and not watermark_mode)
        self.auto_area_button.setText("识别中..." if running else "自动框选")

    @Slot(object)
    def on_inpaint_mode_changed(self, mode):
        fixed_watermark_mode = mode == InpaintMode.FIXED_WATERMARK
        moving_watermark_mode = mode == InpaintMode.MOVING_WATERMARK
        watermark_mode = self._is_watermark_mode(mode)
        self.setting_interface.set_subtitle_controls_enabled(not watermark_mode)
        self.auto_area_button.setEnabled(
            not self._auto_area_button_running and not watermark_mode
        )
        task_index = self.task_list_component.get_current_task_index()
        if task_index < 0:
            return

        if fixed_watermark_mode:
            normalized_areas = self._sanitize_normalized_areas(
                self.task_list_component.get_task_option(
                    task_index, TaskOptions.FIXED_WATERMARK_AREAS, []
                )
            )
            if not normalized_areas:
                normalized_areas = self._parse_fixed_watermark_areas(
                    config.fixedWatermarkSelectionAreas.value
                )
                if normalized_areas:
                    self.task_list_component.update_task_option(
                        task_index,
                        TaskOptions.FIXED_WATERMARK_AREAS,
                        normalized_areas,
                    )
            if normalized_areas:
                self._propagate_fixed_watermark_areas(task_index, normalized_areas)
            self.video_display_component.set_selection_rects(
                self.video_display_component.normalized_video_coordinates_to_preview_coordinates(
                    normalized_areas
                )
            )
            return

        if moving_watermark_mode:
            normalized_area, reference_frame_no, template_source_path = (
                self._moving_watermark_template_for_task(task_index)
            )
            task = self.task_list_component.get_task(task_index)
            if (
                task is not None
                and normalized_area is not None
                and template_source_path
                and os.path.abspath(template_source_path) == os.path.abspath(task.path)
                and reference_frame_no is not None
            ):
                self._seek_preview_to_frame(reference_frame_no)
            self._selection_change_source = "moving_template"
            try:
                self.video_display_component.set_selection_rects(
                    self.video_display_component.normalized_video_coordinates_to_preview_coordinates(
                        [normalized_area] if normalized_area is not None else []
                    )
                )
            finally:
                self._selection_change_source = None
            return

        subtitle_areas = self.task_list_component.get_task_option(
            task_index, TaskOptions.SUB_AREAS, []
        )
        if subtitle_areas:
            self.video_display_component.set_selection_rects(subtitle_areas)
            return
        self._selection_change_source = "default"
        try:
            self.video_display_component.load_selections_from_config()
        finally:
            self._selection_change_source = None

    def _task_needs_auto_area(self, task_index, video_path):
        if self._is_watermark_mode():
            return False
        subtitle_areas = self.task_list_component.get_task_option(task_index, TaskOptions.SUB_AREAS, [])
        subtitle_areas_source = self.task_list_component.get_task_option(task_index, TaskOptions.SUB_AREAS_SOURCE, "")
        return (
            config.autoSubtitleAreaSelection.value
            and not is_image_file(video_path)
            and (not subtitle_areas or subtitle_areas_source in ("", "default", "fallback"))
        )

    def _schedule_auto_area_detection(self, task_index, video_path):
        if not self._task_needs_auto_area(task_index, video_path):
            return

        with self._auto_area_lock:
            if video_path in self._auto_area_results or video_path in self._auto_area_futures:
                return
            self._auto_area_errors.pop(video_path, None)
            future = self.auto_area_executor.submit(auto_detect_subtitle_area, video_path)
            self._auto_area_futures[video_path] = future

        def _done_callback(done_future):
            try:
                result = done_future.result()
            except Exception as e:
                with self._auto_area_lock:
                    self._auto_area_errors[video_path] = str(e)
                    self._auto_area_futures.pop(video_path, None)
                return

            with self._auto_area_lock:
                self._auto_area_results[video_path] = result
                self._auto_area_futures.pop(video_path, None)
                self._auto_area_errors.pop(video_path, None)
            current_task_index = self.task_list_component.find_task_index_by_path(video_path)
            if current_task_index >= 0:
                subtitle_areas_video, _ = result
                self._schedule_subtitle_interval_detection(
                    current_task_index,
                    video_path,
                    subtitle_areas_video,
                    self.task_list_component.get_task_option(current_task_index, TaskOptions.AB_SECTIONS, []),
                )

        future.add_done_callback(_done_callback)

    def _schedule_pending_auto_area_detections(self):
        for task_index, task in self.task_list_component.get_pending_tasks():
            self._schedule_auto_area_detection(task_index, task.path)

    def _get_auto_area_detection_result(self, video_path):
        with self._auto_area_lock:
            result = self._auto_area_results.get(video_path)
            future = self._auto_area_futures.get(video_path)
            error = self._auto_area_errors.get(video_path)
        return result, future, error

    def _task_needs_subtitle_intervals(self, task_index, video_path):
        intervals = self.task_list_component.get_task_option(task_index, TaskOptions.SUBTITLE_INTERVALS, None)
        return (
            config.inpaintMode.value == InpaintMode.STTN_AUTO
            and not is_image_file(video_path)
            and intervals is None
        )

    def _schedule_subtitle_interval_detection(self, task_index, video_path, subtitle_areas_video, ab_sections=None):
        if not subtitle_areas_video or not self._task_needs_subtitle_intervals(task_index, video_path):
            return

        with self._auto_area_lock:
            if video_path in self._subtitle_interval_results or video_path in self._subtitle_interval_futures:
                return
            self._subtitle_interval_errors.pop(video_path, None)
            future = self.auto_area_executor.submit(
                detect_subtitle_intervals,
                video_path,
                subtitle_areas_video,
                ab_sections,
            )
            self._subtitle_interval_futures[video_path] = future

        def _done_callback(done_future):
            try:
                result = done_future.result()
            except Exception as e:
                with self._auto_area_lock:
                    self._subtitle_interval_errors[video_path] = str(e)
                    self._subtitle_interval_futures.pop(video_path, None)
                return

            with self._auto_area_lock:
                self._subtitle_interval_results[video_path] = result
                self._subtitle_interval_futures.pop(video_path, None)
                self._subtitle_interval_errors.pop(video_path, None)

        future.add_done_callback(_done_callback)

    def _schedule_pending_subtitle_interval_detections(self):
        if config.inpaintMode.value != InpaintMode.STTN_AUTO:
            return
        for task_index, task in self.task_list_component.get_pending_tasks():
            result, _, _ = self._get_auto_area_detection_result(task.path)
            if result is None:
                continue
            subtitle_areas_video, _ = result
            self._schedule_subtitle_interval_detection(
                task_index,
                task.path,
                subtitle_areas_video,
                self.task_list_component.get_task_option(task_index, TaskOptions.AB_SECTIONS, []),
            )

    def _get_subtitle_interval_result(self, video_path):
        with self._auto_area_lock:
            result = self._subtitle_interval_results.get(video_path)
            future = self._subtitle_interval_futures.get(video_path)
            error = self._subtitle_interval_errors.get(video_path)
        return result, future, error

    def _apply_detected_areas_to_task(self, task_index, detected_areas, confidence, source="auto", log_prefix=""):
        preview_areas = self.video_display_component.video_coordinates_to_preview_coordinates(detected_areas)
        if not preview_areas:
            if log_prefix:
                self.append_log_signal.emit([f"{log_prefix}自动框选失败: 无法转换字幕区域坐标"])
            return None

        self.video_display_component.set_selection_rects(preview_areas)
        self.video_display_component.save_selections_to_config()
        self.task_list_component.update_task_option(task_index, TaskOptions.SUB_AREAS, preview_areas)
        self.task_list_component.update_task_option(task_index, TaskOptions.SUB_AREAS_SOURCE, source)

        if log_prefix:
            self.append_log_signal.emit([f"{log_prefix}自动框选完成: {detected_areas}, 置信度 {confidence:.2f}"])
        return preview_areas

    def auto_area_button_clicked(self):
        if not self.video_path:
            self.append_output(tr['SubtitleExtractorGUI']['OpenVideoFirst'])
            return
        if self._is_watermark_mode():
            message_key = (
                'MovingWatermarkTemplateRequired'
                if config.inpaintMode.value == InpaintMode.MOVING_WATERMARK
                else 'FixedWatermarkAreaRequired'
            )
            self.append_output(tr['Main'][message_key])
            return

        current_task_index = self.task_list_component.get_current_task_index()
        if current_task_index == -1:
            current_task_index = self.task_list_component.find_task_index_by_path(self.video_path)
            if current_task_index >= 0:
                self.task_list_component.select_task(current_task_index)
            else:
                self.append_output(tr['SubtitleExtractorGUI']['OpenVideoFirst'])
                return

        video_path = self.video_path
        self.append_output("开始自动框选字幕区域...")
        self.auto_subtitle_area_running_signal.emit(True)

        def task():
            try:
                areas, confidence = auto_detect_subtitle_area(video_path)
                self.auto_subtitle_area_signal.emit(areas, confidence)
            except Exception as e:
                traceback.print_exc()
                self.auto_subtitle_area_error_signal.emit(f"自动框选失败: {e}")
            finally:
                self.auto_subtitle_area_running_signal.emit(False)

        threading.Thread(target=task, daemon=True).start()

    @Slot(list, float)
    def on_auto_subtitle_area_detected(self, areas, confidence):
        if self._is_watermark_mode():
            return
        if not areas:
            self.append_output("自动框选失败: 未找到可用字幕区域")
            return

        preview_areas = self.video_display_component.video_coordinates_to_preview_coordinates(areas)
        if not preview_areas:
            self.append_output("自动框选失败: 无法转换字幕区域坐标")
            return

        self.video_display_component.set_selection_rects(preview_areas)
        self.video_display_component.save_selections_to_config()

        current_task_index = self.task_list_component.get_current_task_index()
        if current_task_index >= 0:
            self.task_list_component.update_task_option(current_task_index, TaskOptions.SUB_AREAS, preview_areas)
            self.task_list_component.update_task_option(current_task_index, TaskOptions.SUB_AREAS_SOURCE, "auto")
            self.task_list_component.update_task_option(current_task_index, TaskOptions.SUBTITLE_INTERVALS, None)
            self._schedule_subtitle_interval_detection(current_task_index, self.video_path, areas, self.task_list_component.get_task_option(current_task_index, TaskOptions.AB_SECTIONS, []))

        if confidence <= 0:
            self.append_output(f"未检测到稳定字幕，已使用默认底部区域: {areas}")
        else:
            self.append_output(f"自动框选完成: {areas}, 置信度 {confidence:.2f}")

    def ensure_subtitle_areas_before_run(self, task_index, video_path, inpaint_mode=None):
        inpaint_mode = inpaint_mode or config.inpaintMode.value
        subtitle_areas = self.task_list_component.get_task_option(task_index, TaskOptions.SUB_AREAS, [])
        subtitle_areas_source = self.task_list_component.get_task_option(task_index, TaskOptions.SUB_AREAS_SOURCE, "")
        if inpaint_mode == InpaintMode.FIXED_WATERMARK:
            normalized_areas = self._sanitize_normalized_areas(
                self.task_list_component.get_task_option(
                    task_index, TaskOptions.FIXED_WATERMARK_AREAS, []
                )
            )
            if not normalized_areas:
                raise ValueError(tr['Main']['FixedWatermarkAreaRequired'])
            preview_areas = self.video_display_component.normalized_video_coordinates_to_preview_coordinates(
                normalized_areas
            )
            if not preview_areas:
                raise ValueError(tr['Main']['FixedWatermarkAreaRequired'])
            return preview_areas
        if inpaint_mode == InpaintMode.MOVING_WATERMARK:
            normalized_area, reference_frame_no, _ = self._moving_watermark_template_for_task(
                task_index
            )
            if normalized_area is None or reference_frame_no is None:
                raise ValueError(tr['Main']['MovingWatermarkTemplateRequired'])
            preview_areas = self.video_display_component.normalized_video_coordinates_to_preview_coordinates(
                [normalized_area]
            )
            if len(preview_areas) != 1:
                raise ValueError(tr['Main']['MovingWatermarkTemplateRequired'])
            return preview_areas
        should_auto_detect = self._task_needs_auto_area(task_index, video_path)
        if subtitle_areas and len(subtitle_areas) > 0 and not should_auto_detect:
            return subtitle_areas

        if should_auto_detect:
            try:
                result, future, error = self._get_auto_area_detection_result(video_path)
                if result is None and future is None and error is None:
                    self._schedule_auto_area_detection(task_index, video_path)
                    result, future, error = self._get_auto_area_detection_result(video_path)

                if result is None and future is not None:
                    self.append_log_signal.emit([f"运行前等待自动框选结果: {os.path.basename(video_path)}"])
                    detected_areas, confidence = future.result()
                    with self._auto_area_lock:
                        self._auto_area_results[video_path] = (detected_areas, confidence)
                        self._auto_area_futures.pop(video_path, None)
                        self._auto_area_errors.pop(video_path, None)
                elif result is not None:
                    detected_areas, confidence = result
                else:
                    raise RuntimeError(error or "自动框选失败")

                preview_areas = self._apply_detected_areas_to_task(
                    task_index,
                    detected_areas,
                    confidence,
                    source="auto",
                    log_prefix="运行前",
                )
                if preview_areas:
                    return preview_areas
                self.append_log_signal.emit(["运行前自动框选失败: 未找到可用字幕区域"])
            except Exception as e:
                traceback.print_exc()
                self.append_log_signal.emit([f"运行前自动框选失败: {e}"])

        subtitle_areas = [(0, self.frame_height, 0, self.frame_width)]
        self.task_list_component.update_task_option(task_index, TaskOptions.SUB_AREAS, subtitle_areas)
        self.task_list_component.update_task_option(task_index, TaskOptions.SUB_AREAS_SOURCE, "fallback")
        return subtitle_areas

    def ensure_subtitle_intervals_before_run(self, task_index, video_path, subtitle_areas_video, inpaint_mode=None):
        inpaint_mode = inpaint_mode or config.inpaintMode.value
        if inpaint_mode != InpaintMode.STTN_AUTO or is_image_file(video_path):
            return None

        intervals = self.task_list_component.get_task_option(task_index, TaskOptions.SUBTITLE_INTERVALS, None)
        if intervals is not None:
            return intervals

        ab_sections = self.task_list_component.get_task_option(task_index, TaskOptions.AB_SECTIONS, [])
        try:
            result, future, error = self._get_subtitle_interval_result(video_path)
            if result is None and future is None and error is None:
                self._schedule_subtitle_interval_detection(task_index, video_path, subtitle_areas_video, ab_sections)
                result, future, error = self._get_subtitle_interval_result(video_path)

            if result is None and future is not None:
                self.append_log_signal.emit([f"运行前等待字幕区间分析: {os.path.basename(video_path)}"])
                intervals = future.result()
                with self._auto_area_lock:
                    self._subtitle_interval_results[video_path] = intervals
                    self._subtitle_interval_futures.pop(video_path, None)
                    self._subtitle_interval_errors.pop(video_path, None)
            elif result is not None:
                intervals = result
            else:
                raise RuntimeError(error or "字幕区间分析失败")

            self.task_list_component.update_task_option(task_index, TaskOptions.SUBTITLE_INTERVALS, intervals)
            interval_frame_count = sum(max(0, end - start + 1) for start, end in intervals)
            skip_frame_count = max(0, self.frame_count - interval_frame_count) if self.frame_count else 0
            skip_ratio = (skip_frame_count / self.frame_count * 100) if self.frame_count else 0
            self.append_log_signal.emit([
                f"运行前字幕区间分析完成: {len(intervals)} 个区间, 跳过无字幕帧 {skip_frame_count}/{self.frame_count} ({skip_ratio:.1f}%)"
            ])
            return intervals
        except Exception as e:
            traceback.print_exc()
            self.append_log_signal.emit([f"运行前字幕区间分析失败: {e}"])
            self.task_list_component.update_task_option(task_index, TaskOptions.SUBTITLE_INTERVALS, [])
            return []

    def _is_valid_moving_watermark_task(self, task_index, task):
        if task is None or not is_video_file(task.path) or is_image_file(task.path):
            return False
        normalized_area, reference_frame_no, template_source_path = (
            self._moving_watermark_template_for_task(task_index)
        )
        if normalized_area is None or reference_frame_no is None:
            return False
        template_source_path = template_source_path or task.path
        if not os.path.isfile(template_source_path) or not is_video_file(template_source_path):
            return False
        source_frame_count = self._get_video_frame_count(template_source_path)
        if not 0 <= reference_frame_no < source_frame_count:
            return False
        source_areas = self._normalized_areas_to_video_coordinates(
            [normalized_area],
            self._get_video_dimensions(template_source_path),
        )
        if len(source_areas) != 1:
            return False
        ymin, ymax, xmin, xmax = source_areas[0]
        if ymax - ymin < 8 or xmax - xmin < 8:
            return False
        target_areas = self._normalized_areas_to_video_coordinates(
            [normalized_area],
            self._get_video_dimensions(task.path),
        )
        if len(target_areas) != 1:
            return False
        ymin, ymax, xmin, xmax = target_areas[0]
        return ymax - ymin >= 8 and xmax - xmin >= 8

    def _begin_moving_preprocess_generation(self, run_mode):
        with self._moving_preprocess_lock:
            self._moving_preprocess_cancel_event.set()
            futures_to_cancel = tuple(self._moving_preprocess_futures.values())
            self._moving_preprocess_generation += 1
            self._moving_preprocess_cancel_event = threading.Event()
            self._moving_preprocess_futures.clear()
            self._moving_preprocess_results.clear()
            self._moving_preprocess_errors.clear()
            self._active_run_mode = run_mode
        # Future.cancel() invokes done callbacks synchronously for pending work.
        # Cancel only after publishing the new generation and releasing our lock,
        # otherwise _done_callback would deadlock trying to acquire the same lock.
        for future in futures_to_cancel:
            future.cancel()

    def _cancel_moving_preprocessing(self):
        with self._moving_preprocess_lock:
            self._moving_preprocess_cancel_event.set()
            futures_to_cancel = tuple(self._moving_preprocess_futures.values())
            self._moving_preprocess_generation += 1
            self._moving_preprocess_futures.clear()
            self._moving_preprocess_results.clear()
            self._moving_preprocess_errors.clear()
            self._active_run_mode = None
        for future in futures_to_cancel:
            future.cancel()

    def _moving_preprocess_request_for_task(self, task_index, task):
        if not self._is_valid_moving_watermark_task(task_index, task):
            return None
        normalized_area, reference_frame_no, template_source_path = (
            self._moving_watermark_template_for_task(task_index)
        )
        template_source_path = template_source_path or task.path
        dimensions = self._get_video_dimensions(task.path)
        if not dimensions:
            return None
        width, height = dimensions
        frame_count = self._get_video_frame_count(task.path)
        fps = self._get_video_fps(task.path)
        target_areas = self._normalized_areas_to_video_coordinates(
            [normalized_area],
            dimensions,
        )
        if frame_count <= 0 or fps <= 0 or len(target_areas) != 1:
            return None
        ab_sections = task.options.get(TaskOptions.AB_SECTIONS.value, [])
        artifact_key = build_moving_watermark_preprocess_key(
            task.path,
            template_source_path,
            reference_frame_no,
            normalized_area,
            (height, width),
            frame_count,
            fps,
            ab_sections,
            target_areas[0],
        )
        cache_directory = os.path.join(
            tempfile.gettempdir(),
            "vsr-moving-watermark-preprocess",
        )
        artifact_path = os.path.join(cache_directory, f"{artifact_key}.npz")
        return {
            "key": artifact_key,
            "path": artifact_path,
            "video_path": task.path,
            "template_source_path": template_source_path,
            "reference_frame_no": reference_frame_no,
            "template_area": normalized_area,
            "frame_shape": (height, width),
            "frame_count": frame_count,
            "fps": fps,
            "ab_sections": ab_sections,
            "fallback_target_area": target_areas[0],
        }

    def _next_pending_task_for_preprocess(self, current_index):
        current_task = self.task_list_component.get_task(current_index)
        candidates = [
            (index, task)
            for index, task in self.task_list_component.get_pending_tasks()
            if index != current_index
        ]
        if not candidates:
            return None
        if current_task is not None:
            same_batch = [
                item for item in candidates
                if item[1].batch_id == current_task.batch_id
            ]
            if same_batch:
                return same_batch[0]
        return candidates[0]

    def _schedule_next_moving_watermark_preprocess(self, current_index):
        next_task_item = self._next_pending_task_for_preprocess(current_index)
        if next_task_item is None:
            return
        next_index, next_task = next_task_item
        request = self._moving_preprocess_request_for_task(next_index, next_task)
        if request is None:
            return
        artifact_key = request["key"]
        with self._moving_preprocess_lock:
            if self._active_run_mode != InpaintMode.MOVING_WATERMARK:
                return
            if (
                artifact_key in self._moving_preprocess_results
                or artifact_key in self._moving_preprocess_futures
            ):
                return
            generation = self._moving_preprocess_generation
            cancel_event = self._moving_preprocess_cancel_event
            self._moving_preprocess_errors.pop(artifact_key, None)
            kwargs = {
                key: value
                for key, value in request.items()
                if key not in ("key", "path")
            }
            kwargs["cancel_event"] = cancel_event
            future = self.moving_preprocess_executor.submit(
                preprocess_moving_watermark_to_file,
                request["path"],
                **kwargs,
            )
            self._moving_preprocess_futures[artifact_key] = future
            self._moving_preprocess_artifacts.add(request["path"])

        self.append_log_signal.emit([
            tr['Main']['MovingWatermarkPreprocessScheduled'].format(
                os.path.basename(next_task.path)
            )
        ])

        def _done_callback(done_future):
            try:
                result = done_future.result()
            except Exception as error:
                with self._moving_preprocess_lock:
                    if generation != self._moving_preprocess_generation:
                        return
                    if self._moving_preprocess_futures.get(artifact_key) is not done_future:
                        return
                    self._moving_preprocess_errors[artifact_key] = str(error)
                    self._moving_preprocess_futures.pop(artifact_key, None)
                if not cancel_event.is_set():
                    self.append_log_signal.emit([
                        tr['Main']['MovingWatermarkPreprocessFailed'].format(
                            os.path.basename(next_task.path),
                            error,
                        )
                    ])
                return
            with self._moving_preprocess_lock:
                if generation != self._moving_preprocess_generation or cancel_event.is_set():
                    return
                if self._moving_preprocess_futures.get(artifact_key) is not done_future:
                    return
                self._moving_preprocess_results[artifact_key] = result
                self._moving_preprocess_futures.pop(artifact_key, None)
                self._moving_preprocess_errors.pop(artifact_key, None)
            self.append_log_signal.emit([
                tr['Main']['MovingWatermarkPreprocessReady'].format(
                    os.path.basename(next_task.path),
                    result["elapsed"],
                )
            ])

        future.add_done_callback(_done_callback)

    def _take_moving_watermark_preprocess(self, task_index, task):
        request = self._moving_preprocess_request_for_task(task_index, task)
        if request is None:
            return None
        artifact_key = request["key"]
        with self._moving_preprocess_lock:
            result = self._moving_preprocess_results.get(artifact_key)
            future = self._moving_preprocess_futures.get(artifact_key)
            error = self._moving_preprocess_errors.get(artifact_key)
            generation = self._moving_preprocess_generation
            cancel_event = self._moving_preprocess_cancel_event
        if result is None and future is not None:
            if not future.done():
                self.append_log_signal.emit([
                    tr['Main']['MovingWatermarkPreprocessWaiting'].format(
                        os.path.basename(task.path)
                    )
                ])
            while result is None and not self._stop_event.is_set() and not cancel_event.is_set():
                try:
                    result = future.result(timeout=0.2)
                except FutureTimeoutError:
                    continue
                except Exception as future_error:
                    error = str(future_error)
                    break
        if result is None:
            if error and not cancel_event.is_set():
                self.append_log_signal.emit([
                    tr['Main']['MovingWatermarkPreprocessFailed'].format(
                        os.path.basename(task.path),
                        error,
                    )
                ])
            return None
        with self._moving_preprocess_lock:
            if generation != self._moving_preprocess_generation:
                return None
            self._moving_preprocess_results.pop(artifact_key, None)
            self._moving_preprocess_futures.pop(artifact_key, None)
            self._moving_preprocess_errors.pop(artifact_key, None)
        if result.get("key") != artifact_key or not os.path.isfile(result.get("path", "")):
            return None
        self.append_log_signal.emit([
            tr['Main']['MovingWatermarkPreprocessConsumed'].format(
                os.path.basename(task.path)
            )
        ])
        return result

    @Slot(str, str)
    def on_processing_phase(self, phase, video_path):
        if (
            phase != "inpaint_started"
            or not self._is_processing
            or self._active_run_mode != InpaintMode.MOVING_WATERMARK
            or not config.hardwareAcceleration.value
        ):
            return
        current_index = self.current_processing_task_index
        current_task = self.task_list_component.get_task(current_index)
        if current_task is None or os.path.abspath(current_task.path) != os.path.abspath(video_path):
            return
        self._schedule_next_moving_watermark_preprocess(current_index)

    def run_button_clicked(self):
        run_mode = config.inpaintMode.value
        pending_tasks = self.task_list_component.get_pending_tasks()
        if not pending_tasks:
            self.append_output(tr['SubtitleExtractorGUI']['OpenVideoFirst'])
            return
        if run_mode == InpaintMode.FIXED_WATERMARK:
            missing_selection = [
                (index, task)
                for index, task in pending_tasks
                if not self._sanitize_normalized_areas(
                    task.options.get(TaskOptions.FIXED_WATERMARK_AREAS.value)
                )
            ]
            if missing_selection:
                self.task_list_component.select_task(missing_selection[0][0])
                self.append_output(tr['Main']['FixedWatermarkAreaRequired'])
                return
        if run_mode == InpaintMode.MOVING_WATERMARK:
            invalid_template = [
                (index, task)
                for index, task in pending_tasks
                if not self._is_valid_moving_watermark_task(index, task)
            ]
            if invalid_template:
                self.task_list_component.select_task(invalid_template[0][0])
                self.append_output(tr['Main']['MovingWatermarkTemplateRequired'])
                return

        try:
            # 获取所有待执行的任务
            pending_tasks = self.task_list_component.get_pending_tasks()
            if not pending_tasks:
                return

            self._stop_event.clear()
            self._begin_moving_preprocess_generation(run_mode)
            self._is_processing = True
            self.setting_interface.set_inpaint_mode_enabled(False)
            self.toggle_buttons_signal.emit(False)
            # 开启后台线程处理视频
            def task():
                try:
                    self.append_log_signal.emit(["初始化处理引擎..."])
                    self._ensure_subtitle_worker()
                    if not self._is_watermark_mode(run_mode):
                        self._schedule_pending_auto_area_detections()
                        self._schedule_pending_subtitle_interval_detections()
                    while not self._stop_event.is_set():
                        try:
                            pending_tasks = self.task_list_component.get_pending_tasks()
                            if not pending_tasks:
                                break
                            if not self._is_watermark_mode(run_mode):
                                self._schedule_pending_auto_area_detections()
                                self._schedule_pending_subtitle_interval_detections()
                            current_batch_id = pending_tasks[0][1].batch_id
                            batch_tasks = self.task_list_component.get_pending_tasks_by_batch(current_batch_id)
                            pending_task = batch_tasks[0] if batch_tasks else pending_tasks[0]
                            current_batch_task = pending_task[1]
                            if current_batch_task.source_folder and self.current_processing_batch_id != current_batch_task.batch_id:
                                self.current_processing_batch_id = current_batch_task.batch_id
                                self.append_log_signal.emit([f"\u5904\u7406\u6587\u4ef6\u5939: {current_batch_task.source_folder}"])
                            # 更新当前处理的任务索引
                            self.current_processing_task_index, task_item = pending_task
                            self.current_processing_task_start_time = time.time()
                            if not self.load_video(task_item.path):
                                self.append_log_signal.emit([tr['SubtitleExtractorGUI']['OpenVideoFailed'].format(task_item.path)])
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.FAILED)
                                self.task_list_component.update_task_total_elapsed(
                                    self.current_processing_task_index,
                                    max(0, time.time() - self.current_processing_task_start_time)
                                )
                                self.task_list_component.update_task_process_elapsed(
                                    self.current_processing_task_index,
                                    0
                                )
                                continue

                            tracking_reference_frame_no = None
                            tracking_template_area = None
                            tracking_template_source_path = None
                            # Watermark modes use normalized video coordinates and
                            # do not depend on mutable preview-widget geometry.
                            if run_mode == InpaintMode.FIXED_WATERMARK:
                                normalized_areas = task_item.options.get(
                                    TaskOptions.FIXED_WATERMARK_AREAS.value, []
                                )
                                subtitle_areas_video = self._normalized_areas_to_video_coordinates(
                                    normalized_areas,
                                    self._get_video_dimensions(task_item.path),
                                )
                                if not subtitle_areas_video:
                                    raise ValueError(tr['Main']['FixedWatermarkAreaRequired'])
                            elif run_mode == InpaintMode.MOVING_WATERMARK:
                                (
                                    tracking_template_area,
                                    tracking_reference_frame_no,
                                    tracking_template_source_path,
                                ) = self._moving_watermark_template_for_task(
                                    self.current_processing_task_index
                                )
                                subtitle_areas_video = self._normalized_areas_to_video_coordinates(
                                    [tracking_template_area] if tracking_template_area is not None else [],
                                    self._get_video_dimensions(task_item.path),
                                )
                                if (
                                    len(subtitle_areas_video) != 1
                                    or tracking_reference_frame_no is None
                                ):
                                    raise ValueError(tr['Main']['MovingWatermarkTemplateRequired'])
                                tracking_template_source_path = (
                                    tracking_template_source_path or task_item.path
                                )
                            else:
                                subtitle_areas = self.ensure_subtitle_areas_before_run(
                                    self.current_processing_task_index,
                                    task_item.path,
                                    run_mode,
                                )
                                subtitle_areas_video = self.video_display_component.preview_coordinates_to_video_coordinates(subtitle_areas)
                            subtitle_intervals = self.ensure_subtitle_intervals_before_run(
                                self.current_processing_task_index,
                                task_item.path,
                                subtitle_areas_video,
                                run_mode,
                            )

                            if not self._is_watermark_mode(run_mode):
                                self.video_display_component.save_selections_to_config()

                            # 更新任务状态为运行中
                            self.task_list_component.update_task_progress(self.current_processing_task_index, 1)

                            # 选中当前任务
                            self.select_task_signal.emit(self.current_processing_task_index)

                            with self._video_cap_lock:
                                if self.video_cap:
                                    self.video_cap.release()
                                    self.video_cap = None

                            self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.PROCESSING)
                            options = {}
                            for key in task_item.options:
                                if key in (
                                    TaskOptions.SUB_AREAS_SOURCE.value,
                                    TaskOptions.SUB_AREAS.value,
                                    TaskOptions.FIXED_WATERMARK_AREAS.value,
                                    TaskOptions.MOVING_WATERMARK_TEMPLATE_AREA.value,
                                    TaskOptions.MOVING_WATERMARK_REFERENCE_FRAME_NO.value,
                                    TaskOptions.MOVING_WATERMARK_TEMPLATE_SOURCE_PATH.value,
                                ):
                                    continue
                                if (
                                    self._is_watermark_mode(run_mode)
                                    and key == TaskOptions.SUBTITLE_INTERVALS.value
                                ):
                                    continue
                                options[key] = task_item.options[key]
                            options[TaskOptions.SUB_AREAS.value] = subtitle_areas_video
                            if run_mode == InpaintMode.MOVING_WATERMARK:
                                options["tracking_reference_frame_no"] = tracking_reference_frame_no
                                options["tracking_template_area"] = tracking_template_area
                                options["tracking_template_source_path"] = tracking_template_source_path
                                preprocess_result = self._take_moving_watermark_preprocess(
                                    self.current_processing_task_index,
                                    task_item,
                                )
                                if preprocess_result is not None:
                                    options["moving_watermark_preprocess_path"] = preprocess_result["path"]
                            if subtitle_intervals is not None:
                                options[TaskOptions.SUBTITLE_INTERVALS.value] = subtitle_intervals
                            options["inpaint_mode"] = run_mode
                            # 清理缓存, 使用动态路径
                            task_item.output_path = None
                            output_path = task_item.output_path
                            os.makedirs(os.path.dirname(output_path), exist_ok=True)
                            self.current_processing_run_start_time = time.time()
                            process = self.run_subtitle_remover_process(task_item.path, output_path, options)

                            # 检查是否在处理过程中被停止
                            if self._stop_event.is_set():
                                break

                            # 更新任务状态为已完成
                            task_obj = self.task_list_component.get_task(self.current_processing_task_index)
                            if self.last_worker_job_succeeded and task_obj and task_obj.status == TaskStatus.PROCESSING:
                                self.task_list_component.update_task_total_elapsed(
                                    self.current_processing_task_index,
                                    max(0, time.time() - self.current_processing_task_start_time)
                                )
                                self.task_list_component.update_task_process_elapsed(
                                    self.current_processing_task_index,
                                    max(0, time.time() - self.current_processing_run_start_time) if self.current_processing_run_start_time else 0
                                )
                                self.progress_signal.emit(100, True)
                                # 任务完成, 更新输出路径为只读
                                task_obj.output_path = output_path
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.COMPLETED)
                            else:
                                self.task_list_component.update_task_total_elapsed(
                                    self.current_processing_task_index,
                                    max(0, time.time() - self.current_processing_task_start_time)
                                )
                                self.task_list_component.update_task_process_elapsed(
                                    self.current_processing_task_index,
                                    max(0, time.time() - self.current_processing_run_start_time) if self.current_processing_run_start_time else 0
                                )
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.FAILED)

                        except Exception as e:
                            print(e)
                            self.append_log_signal.emit([f"Error: {e}"])
                            # 更新任务状态为失败
                            if self.current_processing_task_index >= 0:
                                self.task_list_component.update_task_total_elapsed(
                                    self.current_processing_task_index,
                                    max(0, time.time() - self.current_processing_task_start_time) if self.current_processing_task_start_time else 0
                                )
                                self.task_list_component.update_task_process_elapsed(
                                    self.current_processing_task_index,
                                    max(0, time.time() - self.current_processing_run_start_time) if self.current_processing_run_start_time else 0
                                )
                                self.task_status_signal.emit(self.current_processing_task_index, TaskStatus.FAILED)
                            break
                        finally:
                            self.current_processing_task_start_time = None
                            self.current_processing_run_start_time = None
                            with self._video_cap_lock:
                                if self.video_cap:
                                    self.video_cap.release()
                                    self.video_cap = None
                finally:
                    self._cancel_moving_preprocessing()
                    if config.inpaintMode.value != run_mode:
                        config.set(config.inpaintMode, run_mode)
                    self._saved_inpaint_mode = None
                    self.toggle_buttons_signal.emit(True)

            self._worker_thread = threading.Thread(target=task, daemon=True)
            self._worker_thread.start()
        except Exception as e:
            self._cancel_moving_preprocessing()
            self._is_processing = False
            self.setting_interface.set_inpaint_mode_enabled(self._saved_inpaint_mode is None)
            print(traceback.format_exc())
            self.append_log_signal.emit([f"Error: {e}"])
            self.toggle_buttons_signal.emit(True)

    @staticmethod
    def remover_process(queue, video_path, output_path, options):
        """
        在子进程中执行字幕提取的函数
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            options: 选项
        """
        sr = None
        try:
            from backend.main import SubtitleRemover
            options = dict(options)
            inpaint_mode = options.pop("inpaint_mode", None)
            if inpaint_mode is not None:
                config.inpaintMode.value = inpaint_mode
            sr = SubtitleRemover(video_path, True)
            sr.video_out_path = output_path
            for key in options:
                setattr(sr, key, options[key])
            sr.add_progress_listener(lambda progress, isFinished: SubtitleRemoverRemoteCall.remote_call_update_progress(queue, progress, isFinished))
            sr.append_output = lambda *args: SubtitleRemoverRemoteCall.remote_call_append_log(queue, args)
            sr.manage_process = lambda pid: SubtitleRemoverRemoteCall.remote_call_manage_process(queue, pid)
            sr.update_preview_with_comp = lambda *args: SubtitleRemoverRemoteCall.remote_call_update_preview_with_comp(queue, args)
            sr.report_processing_phase = lambda phase, path: SubtitleRemoverRemoteCall.remote_call_processing_phase(
                queue,
                phase,
                path,
            )
            sr.run()
        except Exception as e:
            traceback.print_exc()
            SubtitleRemoverRemoteCall.remote_call_catch_error(queue, e)
        finally:
            if sr:
                sr.isFinished = True
                sr.vsf_running = False
            SubtitleRemoverRemoteCall.remote_call_finish_job(queue)
            

    # 修改run_subtitle_remover_process方法
    @staticmethod
    def remover_worker_process(command_queue, queue):
        while True:
            command = command_queue.get()
            if not command:
                continue
            action = command[0]
            if action == "shutdown":
                break
            if action != "run" or len(command) != 4:
                continue
            _, video_path, output_path, options = command
            HomeInterface.remover_process(queue, video_path, output_path, options)

    def run_subtitle_remover_process(self, video_path, output_path, options):
        """
        使用多进程执行字幕提取，并等待进程完成
        
        Args:
            video_path: 视频文件路径
            output_path: 输出文件路径
            options: 任务选项
        """
        process = self._ensure_subtitle_worker()
        remote_caller = self.worker_remote_caller
        self.last_worker_job_succeeded = False
        if self._stop_event.is_set() or remote_caller is None or self.worker_command_queue is None:
            return process

        remote_caller.reset_job_state()
        self.worker_command_queue.put(("run", video_path, output_path, options))
        self.running_process = process

        while not self._stop_event.is_set():
            if remote_caller.wait_for_job_finish(timeout=0.2):
                self.last_worker_job_succeeded = not remote_caller.job_had_error
                break
            if not process.is_alive():
                break

        if not process.is_alive():
            print(f"Worker process exited with code {process.exitcode}")
            if not self._stop_event.is_set():
                self._dispose_subtitle_worker(terminate=False)
        return process

    @Slot()
    def processing_finished(self):
        pending_tasks = self.task_list_component.get_pending_tasks()
        if pending_tasks:
            # 还有待执行任务, 忽略
            return
        # 处理完成后恢复界面可用性
        self.run_button.setVisible(True)
        self.stop_button.setVisible(False)
        self.se = None
        self.current_processing_batch_id = None
        # 重置视频滑块
        self.video_slider.setValue(1)
        # 重置当前处理任务索引
        self.current_processing_task_index = -1

    @Slot(int, bool)
    def update_progress(self, progress_total, isFinished):
        try:
            pos = min(self.frame_count - 1, int(progress_total / 100 * self.frame_count))
            if pos != self.video_slider.value():
                self.video_slider.blockSignals(True)
                self.video_slider.setValue(pos)
                self.video_slider.blockSignals(False)
            
            # 更新任务进度
            if self.current_processing_task_index >= 0:
                self.task_list_component.update_task_progress(
                    self.current_processing_task_index, 
                    progress_total,
                )
            
            # 检查是否完成
            if isFinished:
                self.processing_finished()
        except Exception as e:
            # 捕获任何异常，防止崩溃
            print(f"更新进度时出错: {str(e)}")

    @Slot(list)
    def append_log(self, log):
        self.append_output(*log)

    def append_output(self, *args):
        """添加文本到输出区域并控制滚动
        Args:
            *args: 要输出的内容，多个参数将用空格连接
        """
        # 将所有参数转换为字符串并用空格连接
        text = ' '.join(str(arg) for arg in args).rstrip()
        timestamp = datetime.now().strftime('%H:%M:%S')
        # 转义HTML特殊字符
        escaped = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        # 根据内容判断消息类型并着色
        if '错误' in text or 'Error' in text or '失败' in text or 'Failed' in text:
            color = '#e74c3c'
        elif '成功' in text or '完成' in text or 'Success' in text or 'Finished' in text:
            color = '#27ae60'
        elif '警告' in text or 'Warning' in text:
            color = '#f39c12'
        else:
            color = '#2980b9'
        html = f'<span style="color:#888;">[{timestamp}]</span> <span style="color:{color};">{escaped}</span><br>'
        self.output_text.append(html)
        print(*args)  # 保持原始的 print 行为
        # 如果启用了自动滚动，则滚动到底部
        if self.auto_scroll:
            scrollbar = self.output_text.verticalScrollBar()
            scrollbar.setValue(scrollbar.maximum())

    @Slot(list)
    def update_preview_with_comp(self, args):
        """更新执行时预览"""
        frame_ori, frame_comp = args
        if self.current_processing_task_index >= 0:
            subtitle_areas = self.task_list_component.get_task_option(self.current_processing_task_index, TaskOptions.SUB_AREAS, [])
            if len(subtitle_areas) > 0:
                subtitle_areas = self.video_display_component.preview_coordinates_to_video_coordinates(subtitle_areas)
                if frame_ori is frame_comp:
                    frame_ori = frame_ori.copy()
                for rect in subtitle_areas:
                    ymin, ymax, xmin, xmax = rect
                    cv2.rectangle(frame_ori, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
        preview_frame = cv2.hconcat([frame_ori, frame_comp])
        # 先缩放图像
        resized_frame = self._img_resize(preview_frame)
        # 更新视频显示（这会同时保存current_pixmap）
        self.video_display_component.update_video_display(resized_frame, draw_selection=False)
        self.video_display_component.set_dragger_enabled(False)

    @Slot(object)
    def on_task_error(self, e):
        self.append_output(tr['SubtitleExtractorGUI']['ErrorDuringProcessing'].format(str(e)))
        if self.current_processing_task_index >= 0:
            self.task_list_component.update_task_status(self.current_processing_task_index, TaskStatus.FAILED)

    def load_video(self, video_path):
        self.video_path = video_path
        with self._video_cap_lock:
            if self.video_cap:
                self.video_cap.release()
                self.video_cap = None
        # 如果是图片文件，直接走图片加载路径
        if is_image_file(video_path):
            return self.load_as_picture(video_path)
        with self._video_cap_lock:
            self.video_cap = cv2.VideoCapture(get_readable_path(self.video_path))
            if not self.video_cap.isOpened():
                self.video_cap = None
                return self.load_as_picture(video_path)
            ret, frame = self.video_cap.read()
            if not ret:
                self.video_cap.release()
                self.video_cap = None
                return self.load_as_picture(video_path)
            self.frame_count = int(self.video_cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.frame_height = int(self.video_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_width = int(self.video_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.fps = self.video_cap.get(cv2.CAP_PROP_FPS)
            self._displayed_frame_no = 0
            self._video_dimension_cache[video_path] = (self.frame_width, self.frame_height)
            self._video_frame_count_cache[video_path] = self.frame_count

        self.update_preview(frame)
        self.video_slider.setMaximum(max(1, self.frame_count - 1))
        self.video_slider.setValue(1)
        self.video_display_component.set_dragger_enabled(not self._is_processing)
        # 视频模式下恢复用户原始的 inpaint 模式选择
        self._unlock_inpaint_mode()
        return True

    def load_as_picture(self, path):
        if not is_image_file(path):
            return False
        self.video_path = path
        self.video_cap = None
        frame = read_image(get_readable_path(path))
        if frame is None:
            return False
        self.frame_count = 1
        self.frame_height = frame.shape[0]
        self.frame_width = frame.shape[1]
        self._displayed_frame_no = 0
        self._video_dimension_cache[path] = (self.frame_width, self.frame_height)
        self._video_frame_count_cache[path] = self.frame_count
        self.fps = 1
        self.update_preview(frame)
        self.video_slider.setMaximum(self.frame_count)
        self.video_slider.setValue(1)
        self.video_display_component.set_dragger_enabled(not self._is_processing)
        # 图片模式锁定为 LAMA
        self._lock_inpaint_mode_to_lama()
        return True

    def _lock_inpaint_mode_to_lama(self):
        """图片模式锁定 inpaint 模式为 LAMA"""
        if self._saved_inpaint_mode is None:
            self._saved_inpaint_mode = config.inpaintMode.value
        config.set(config.inpaintMode, InpaintMode.LAMA)
        self.setting_interface.set_inpaint_mode_enabled(False)

    def _unlock_inpaint_mode(self):
        """视频模式恢复用户原始的 inpaint 模式选择"""
        if self._saved_inpaint_mode is not None:
            config.set(config.inpaintMode, self._saved_inpaint_mode)
            self._saved_inpaint_mode = None
        self.setting_interface.set_inpaint_mode_enabled(not self._is_processing)
        self.video_slider.setValue(1)
        self.video_display_component.set_dragger_enabled(not self._is_processing)
        return True


    def _build_folder_output_subdirs(self, folders):
        counter = Counter()
        subdirs = {}
        for folder in folders:
            folder_name = os.path.basename(os.path.normpath(folder)) or "output"
            counter[folder_name] += 1
            subdirs[folder] = folder_name if counter[folder_name] == 1 else f"{folder_name}_{counter[folder_name]}"
        return subdirs

    def _collect_supported_files(self, folder):
        files = []
        for name in os.listdir(folder):
            path = os.path.join(folder, name)
            if not os.path.isfile(path) or not is_video_file(path):
                continue
            stem = os.path.splitext(os.path.basename(path))[0].lower()
            if stem.endswith("_no_sub"):
                continue
            files.append(path)
        return sorted(files, key=_episode_name_sort_key)

    def open_folders_batch(self):
        output_root = config.saveDirectory.value or ""
        if not output_root:
            self.append_output("\u8bf7\u5148\u5728\u9ad8\u7ea7\u8bbe\u7f6e -> \u89c6\u9891\u4fdd\u5b58\u76ee\u5f55 \u4e2d\u9009\u62e9\u8f93\u51fa\u6839\u76ee\u5f55")
            return

        dialog = BatchFolderManagerDialog(
            self,
            default_output_root=output_root
        )
        dialog.output_root = output_root
        dialog.output_root_edit.setText(output_root)
        dialog.output_root_edit.setEnabled(False)
        dialog.choose_output_button.setEnabled(False)
        if dialog.exec() != QtWidgets.QDialog.Accepted:
            return
        folders = dialog.selected_folders

        folder_output_subdirs = self._build_folder_output_subdirs(folders)
        imported_folder_count = 0
        imported_file_count = 0
        first_task_index = -1

        for folder in folders:
            files = self._collect_supported_files(folder)
            if not files:
                self.append_output(f"\u6587\u4ef6\u5939\u5185\u672a\u627e\u5230\u652f\u6301\u7684\u89c6\u9891\u6587\u4ef6: {folder}")
                continue

            imported_folder_count += 1
            batch_id = folder
            output_subdir = folder_output_subdirs[folder]

            for path in files:
                if self.load_video(path):
                    self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoSuccess']}: {path}")
                    row = self.task_list_component.add_task(
                        path,
                        batch_id=batch_id,
                        source_folder=folder,
                        output_root=output_root,
                        output_subdir=output_subdir,
                    )
                    index = row if isinstance(row, int) and row >= 0 else max(0, self.task_list_component.find_task_index_by_path(path))
                    task = self.task_list_component.get_task(index)
                    if task and task.status == TaskStatus.COMPLETED:
                        self.append_output(f"检测到已处理完成，跳过重复执行: {path}")
                    if first_task_index == -1:
                        first_task_index = index
                    imported_file_count += 1
                else:
                    self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoFailed']}: {path}")

        if first_task_index >= 0:
            self.task_list_component.select_task(first_task_index)
            self.append_output(
                f"\u6279\u91cf\u5bfc\u5165\u5b8c\u6210: {imported_folder_count} \u4e2a\u6587\u4ef6\u5939, {imported_file_count} \u4e2a\u89c6\u9891/\u56fe\u7247, \u8f93\u51fa\u6839\u76ee\u5f55 {output_root}"
            )

    def open_file(self):
        files, _ = QtWidgets.QFileDialog.getOpenFileNames(
            self,
            tr['SubtitleExtractorGUI']['Open'],
            "",
            "All Files (*.*);;Video Files (*.mp4 *.flv *.wmv *.avi *.mkv *.mov);;Image Files (*.jpg *.jpeg *.png *.bmp *.webp *.tiff)"
        )
        if files:
            files_loaded = []
            # 倒序打开, 确保第一个视频截图显示在屏幕上
            for path in reversed(files):
                if self.load_video(path):
                    self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoSuccess']}: {path}")
                    files_loaded.append(path)
                else:
                    self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoFailed']}: {path}")
            # 正序添加, 确保任务列表顺序一致
            for path in reversed(files_loaded):
                # 添加到任务列表
                row = self.task_list_component.add_task(path, batch_id=path)
                index = row if isinstance(row, int) and row >= 0 else max(0, self.task_list_component.find_task_index_by_path(path))
                task = self.task_list_component.get_task(index)
                if task and task.status == TaskStatus.COMPLETED:
                    self.append_output(f"检测到已处理完成，跳过重复执行: {path}")
                self.task_list_component.select_task(index)

    def open_folder(self):
        folder = QtWidgets.QFileDialog.getExistingDirectory(
            self,
            "\u6253\u5f00\u6587\u4ef6\u5939",
            ""
        )
        if not folder:
            return

        files = self._collect_supported_files(folder)
        if not files:
            self.append_output(f"\u6587\u4ef6\u5939\u5185\u672a\u627e\u5230\u652f\u6301\u7684\u89c6\u9891\u6587\u4ef6: {folder}")
            return

        files_loaded = []
        for path in reversed(files):
            if self.load_video(path):
                self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoSuccess']}: {path}")
                files_loaded.append(path)
            else:
                self.append_output(f"{tr['SubtitleExtractorGUI']['OpenVideoFailed']}: {path}")

        output_root = config.saveDirectory.value or ""
        output_subdir = self._build_folder_output_subdirs([folder])[folder] if output_root else ""

        for path in reversed(files_loaded):
            row = self.task_list_component.add_task(
                path,
                batch_id=folder,
                source_folder=folder,
                output_root=output_root,
                output_subdir=output_subdir,
            )
            index = row if isinstance(row, int) and row >= 0 else max(0, self.task_list_component.find_task_index_by_path(path))
            task = self.task_list_component.get_task(index)
            if task and task.status == TaskStatus.COMPLETED:
                self.append_output(f"检测到已处理完成，跳过重复执行: {path}")
            self.task_list_component.select_task(index)

    def closeEvent(self, event):
        """窗口关闭时断开信号连接并清理资源"""
        try:
            # 通知 worker 线程停止
            self._stop_event.set()
            self._cancel_moving_preprocessing()
            # 终止子进程
            self._dispose_subtitle_worker(terminate=False)
            ProcessManager.instance().terminate_all()
            # 等待 worker 线程结束（最多5秒）
            if self._worker_thread and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=5)

            # 断开信号连接
            self.progress_signal.disconnect(self.update_progress)
            self.append_log_signal.disconnect(self.append_log)
            self.update_preview_with_comp_signal.disconnect(self.update_preview_with_comp)
            self.task_error_signal.disconnect(self.on_task_error)
            self.toggle_buttons_signal.disconnect(self._toggle_buttons)
            self.processing_phase_signal.disconnect(self.on_processing_phase)
            self.video_display_component.video_slider.valueChanged.disconnect(self.slider_changed)
            self.video_display_component.ab_sections_changed.disconnect(self.ab_sections_changed)
            self.video_display_component.selections_changed.disconnect(self.selections_changed)
            self.auto_area_executor.shutdown(wait=False, cancel_futures=False)
            self.moving_preprocess_executor.shutdown(wait=False, cancel_futures=True)
            for artifact_path in list(self._moving_preprocess_artifacts):
                try:
                    if os.path.isfile(artifact_path):
                        os.remove(artifact_path)
                except OSError:
                    pass
            # 释放视频资源
            with self._video_cap_lock:
                if self.video_cap:
                    self.video_cap.release()
                    self.video_cap = None
        except Exception as e:
            print(f"Error during close window:", e)
        super().closeEvent(event)
    
