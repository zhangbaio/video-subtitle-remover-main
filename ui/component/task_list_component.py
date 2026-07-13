import os
from enum import Enum, unique
from dataclasses import dataclass
from functools import cached_property

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QAction, QColor, QBrush
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHeaderView,
    QMenu,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)
from qfluentwidgets import InfoBar, TableWidget
from showinfm import show_in_file_manager

from backend.config import config, tr
from backend.tools.common_tools import is_image_file


@unique
class TaskStatus(Enum):
    PENDING = tr["TaskList"]["Pending"]
    PROCESSING = tr["TaskList"]["Processing"]
    COMPLETED = tr["TaskList"]["Completed"]
    FAILED = tr["TaskList"]["Failed"]


@unique
class TaskOptions(Enum):
    AB_SECTIONS = "ab_sections"
    SUB_AREAS = "sub_areas"
    SUB_AREAS_SOURCE = "sub_areas_source"
    SUBTITLE_INTERVALS = "subtitle_intervals"
    FIXED_WATERMARK_AREAS = "fixed_watermark_areas"
    MOVING_WATERMARK_TEMPLATE_AREA = "moving_watermark_template_area"
    MOVING_WATERMARK_REFERENCE_FRAME_NO = "moving_watermark_reference_frame_no"
    MOVING_WATERMARK_TEMPLATE_SOURCE_PATH = "moving_watermark_template_source_path"


def _format_elapsed(seconds):
    if seconds <= 0:
        return ""
    if seconds >= 60:
        return f"{seconds / 60:.1f}分钟"
    return f"{int(round(seconds))}秒"


@dataclass
class Task:
    path: str
    name: str
    progress: int
    status: TaskStatus
    options: dict
    batch_id: str = ""
    source_folder: str = ""
    output_root: str = ""
    output_subdir: str = ""
    total_elapsed_seconds: float = 0.0
    process_elapsed_seconds: float = 0.0
    _output_path: str = None

    @property
    def output_path(self):
        if self._output_path is not None:
            return self._output_path
        save_directory = self.output_root or config.saveDirectory.value or os.path.dirname(self.path)
        if self.output_subdir:
            save_directory = os.path.join(save_directory, self.output_subdir)
        source_directory = os.path.dirname(self.path)
        if os.path.abspath(save_directory) == os.path.abspath(source_directory):
            save_directory = os.path.join(save_directory, "no_sub_output")
        return os.path.abspath(os.path.join(save_directory, os.path.basename(self.path)))

    @output_path.setter
    def output_path(self, value):
        self._output_path = value

    @property
    def display_name(self):
        if self.source_folder:
            folder_name = os.path.basename(os.path.normpath(self.source_folder)) or self.source_folder
            return f"{folder_name} / {self.name}"
        return self.name

    @property
    def total_elapsed_display(self):
        return _format_elapsed(self.total_elapsed_seconds)

    @property
    def process_elapsed_display(self):
        return _format_elapsed(self.process_elapsed_seconds)

    @cached_property
    def is_image(self):
        return is_image_file(self.path)


class TaskListComponent(QWidget):
    task_selected = Signal(int, str)
    task_deleted = Signal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("TaskListComponent")
        self.tasks = []
        self.current_task_index = -1
        self.__init_widgets()

    def __init_widgets(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.table = TableWidget(self)
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels([
            tr["TaskList"]["Name"],
            tr["TaskList"]["Progress"],
            tr["TaskList"]["Status"],
            "总耗时",
            "去字幕耗时",
        ])
        self.table.setShowGrid(False)
        self.table.setAlternatingRowColors(True)

        header = self.table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(4, QHeaderView.ResizeToContents)

        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setContextMenuPolicy(Qt.CustomContextMenu)
        self.table.setTextElideMode(Qt.ElideMiddle)

        self.table.customContextMenuRequested.connect(self.show_context_menu)
        self.table.clicked.connect(self.on_task_clicked)
        layout.addWidget(self.table)

    @staticmethod
    def _output_file_exists(output_path):
        return bool(output_path) and os.path.isfile(output_path) and os.path.getsize(output_path) > 0

    def _status_brush(self, status):
        if status == TaskStatus.COMPLETED:
            return QBrush(QColor("#2ecc71"))
        if status == TaskStatus.PROCESSING:
            return QBrush(QColor("#3498db"))
        if status == TaskStatus.FAILED:
            return QBrush(QColor("#e74c3c"))
        return QBrush(QColor())

    def _sync_task_row(self, row):
        if not (0 <= row < len(self.tasks)):
            return
        task = self.tasks[row]
        values = [
            task.display_name,
            f"{task.progress}%",
            task.status.value,
            task.total_elapsed_display,
            task.process_elapsed_display,
        ]
        for col, value in enumerate(values):
            item = self.table.item(row, col)
            if item:
                item.setText(value)

        name_item = self.table.item(row, 0)
        if name_item:
            name_item.setToolTip(task.path)
            name_item.setTextAlignment(Qt.AlignVCenter | Qt.AlignLeft)

        for col in (1, 2, 3, 4):
            item = self.table.item(row, col)
            if item:
                item.setTextAlignment(Qt.AlignCenter)

        status_item = self.table.item(row, 2)
        if status_item:
            status_item.setForeground(self._status_brush(task.status))

    def _sync_task_completion_from_output(self, row):
        if not (0 <= row < len(self.tasks)):
            return
        task = self.tasks[row]
        task._output_path = None
        output_path = task.output_path
        if self._output_file_exists(output_path):
            task.output_path = output_path
            task.progress = 100
            task.status = TaskStatus.COMPLETED
        self._sync_task_row(row)

    def add_task(self, video_path, batch_id="", source_folder="", output_root="", output_subdir=""):
        row = self.find_task_index_by_path(video_path)
        if row >= 0:
            task = self.tasks[row]
            task.name = os.path.basename(video_path)
            task.batch_id = batch_id
            task.source_folder = source_folder
            task.output_root = output_root
            task.output_subdir = output_subdir
            self._sync_task_completion_from_output(row)
            return row

        task = Task(
            path=video_path,
            name=os.path.basename(video_path),
            progress=0,
            status=TaskStatus.PENDING,
            options={},
            batch_id=batch_id,
            source_folder=source_folder,
            output_root=output_root,
            output_subdir=output_subdir,
        )
        self.tasks.append(task)

        row = len(self.tasks) - 1
        self.table.setRowCount(len(self.tasks))
        for col, text in enumerate([
            task.display_name,
            "0%",
            TaskStatus.PENDING.value,
            task.total_elapsed_display,
            task.process_elapsed_display,
        ]):
            item = QTableWidgetItem(text)
            self.table.setItem(row, col, item)

        self._sync_task_completion_from_output(row)
        self.table.scrollToBottom()
        return row

    def update_task_progress(self, index, progress):
        if 0 <= index < len(self.tasks):
            self.tasks[index].progress = progress
            progress_item = self.table.item(index, 1)
            if progress_item:
                progress_item.setText(f"{progress}%")
            if index == self.current_task_index:
                self.table.scrollTo(self.table.model().index(index, 0))

    def update_task_status(self, index, status):
        if 0 <= index < len(self.tasks):
            self.tasks[index].status = status
            self._sync_task_row(index)
            if index == self.current_task_index:
                self.table.scrollTo(self.table.model().index(index, 0))
            self.table.selectRow(index)

    def update_task_elapsed(self, index, elapsed_seconds):
        self.update_task_total_elapsed(index, elapsed_seconds)

    def update_task_total_elapsed(self, index, elapsed_seconds):
        if 0 <= index < len(self.tasks):
            self.tasks[index].total_elapsed_seconds = elapsed_seconds
            item = self.table.item(index, 3)
            if item:
                item.setText(self.tasks[index].total_elapsed_display)

    def update_task_process_elapsed(self, index, elapsed_seconds):
        if 0 <= index < len(self.tasks):
            self.tasks[index].process_elapsed_seconds = elapsed_seconds
            item = self.table.item(index, 4)
            if item:
                item.setText(self.tasks[index].process_elapsed_display)

    def get_pending_tasks(self):
        return [(i, task) for i, task in enumerate(self.tasks) if task.status == TaskStatus.PENDING]

    def get_pending_tasks_by_batch(self, batch_id):
        return [
            (i, task) for i, task in enumerate(self.tasks)
            if task.status == TaskStatus.PENDING and task.batch_id == batch_id
        ]

    def get_all_tasks(self):
        return self.tasks

    def get_task(self, index):
        if 0 <= index < len(self.tasks):
            return self.tasks[index]
        return None

    def find_task_index_by_path(self, path):
        for idx, task in enumerate(self.tasks):
            if task.path == path:
                return idx
        return -1

    def show_context_menu(self, pos):
        index = self.table.indexAt(pos)
        if not index.isValid():
            return

        menu = QMenu(self)

        open_video_location_action = QAction(tr["TaskList"]["OpenSourceVideoLocation"], self)
        open_video_location_action.triggered.connect(
            lambda: self.open_file_location(self.tasks[index.row()].path)
        )
        menu.addAction(open_video_location_action)

        def open_target_location():
            task = self.tasks[index.row()]
            path = task.output_path
            if task.status != TaskStatus.COMPLETED:
                InfoBar.warning(
                    title=tr["TaskList"]["Warning"],
                    content=tr["TaskList"]["TargetFileNotFound"],
                    parent=self.get_root_parent(),
                    duration=3000,
                )
                return
            self.open_file_location(path)

        open_target_location_action = QAction(tr["TaskList"]["OpenTargetVideoLocation"], self)
        open_target_location_action.triggered.connect(open_target_location)
        menu.addAction(open_target_location_action)

        reset_task_status_action = QAction(tr["TaskList"]["ResetTaskStatus"], self)
        reset_task_status_action.triggered.connect(
            lambda: (
                self.update_task_status(index.row(), TaskStatus.PENDING),
                self.update_task_progress(index.row(), 0),
                self.update_task_total_elapsed(index.row(), 0),
                self.update_task_process_elapsed(index.row(), 0),
            )
        )
        menu.addAction(reset_task_status_action)

        delete_action = QAction(tr["TaskList"]["DeleteTask"], self)
        delete_action.triggered.connect(lambda: self.delete_task(index.row()))
        menu.addAction(delete_action)

        menu.exec_(self.table.viewport().mapToGlobal(pos))

    def delete_task(self, row):
        if 0 <= row < len(self.tasks):
            del self.tasks[row]
            self.table.removeRow(row)
            if row == self.current_task_index:
                self.current_task_index = -1
            self.task_deleted.emit(row)

    def clear_tasks(self):
        """Remove every imported task and reset the table selection."""
        cleared_count = len(self.tasks)
        self.tasks.clear()
        self.current_task_index = -1
        self.table.clearSelection()
        self.table.setRowCount(0)
        return cleared_count

    def on_task_clicked(self, index):
        row = index.row()
        if 0 <= row < len(self.tasks):
            self.current_task_index = row
            self.task_selected.emit(row, self.tasks[row].path)

    def set_current_task(self, index):
        if 0 <= index < len(self.tasks):
            self.current_task_index = index
            self.table.selectRow(index)
            self.table.scrollTo(self.table.model().index(index, 0))

    def get_current_task_index(self):
        return self.current_task_index

    def select_task(self, index):
        self.set_current_task(index)
        if 0 <= index < len(self.tasks):
            self.task_selected.emit(index, self.tasks[index].path)

    def open_file_location(self, path):
        if not os.path.exists(path):
            InfoBar.warning(
                title=tr["TaskList"]["Warning"],
                content=tr["TaskList"]["UnableToLocateFile"],
                parent=self.get_root_parent(),
                duration=3000,
            )
            return
        show_in_file_manager(os.path.abspath(path))

    def get_root_parent(self):
        parent = self
        while parent.parent():
            parent = parent.parent()
        return parent

    def update_task_option(self, index, task_option: TaskOptions, value):
        if 0 <= index < len(self.tasks):
            self.tasks[index].options[task_option.value] = value

    def get_task_option(self, index, task_option: TaskOptions, default=None):
        if 0 <= index < len(self.tasks):
            return self.tasks[index].options.get(task_option.value, default)
        return default

    def get_batch_auto_detected_areas(self, batch_id):
        if not batch_id:
            return None
        for task in self.tasks:
            if (
                task.batch_id == batch_id
                and task.options.get(TaskOptions.SUB_AREAS_SOURCE.value) == "auto"
                and task.options.get(TaskOptions.SUB_AREAS.value)
            ):
                return task.options.get(TaskOptions.SUB_AREAS.value)
        return None
