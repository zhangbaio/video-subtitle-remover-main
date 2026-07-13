import os
from pathlib import Path
import sys
import tempfile
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from PySide6.QtWidgets import QApplication

    from backend.config import config
    from backend.tools.constant import InpaintMode
    from ui.component.task_list_component import TaskListComponent, TaskStatus

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


@unittest.skipUnless(GUI_AVAILABLE, "PySide6/qfluentwidgets is not installed")
class TaskCompletionMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.original_mode = config.inpaintMode.value
        self.temporary_directory = tempfile.TemporaryDirectory()
        root = Path(self.temporary_directory.name)
        self.source_path = root / "source.mp4"
        self.output_root = root / "output"
        self.output_path = self.output_root / self.source_path.name
        self.source_path.write_bytes(b"source")
        self.output_root.mkdir()
        self.output_path.write_bytes(b"legacy-output")
        config.set(config.inpaintMode, InpaintMode.SUBTITLE_FIXED_WATERMARK)

    def tearDown(self):
        config.set(config.inpaintMode, self.original_mode)
        self.temporary_directory.cleanup()

    def _add_task(self, component):
        return component.add_task(
            str(self.source_path),
            output_root=str(self.output_root),
        )

    def test_combined_mode_requires_matching_completion_metadata(self):
        component = TaskListComponent()
        index = self._add_task(component)

        # A legacy same-name output predates the combined modes and must not
        # make a new combined job disappear from the pending queue.
        self.assertIs(component.get_task(index).status, TaskStatus.PENDING)

        component.update_task_status(index, TaskStatus.COMPLETED)
        metadata_path = component._completion_metadata_path(str(self.output_path))
        self.assertTrue(os.path.isfile(metadata_path))

        reloaded = TaskListComponent()
        reloaded_index = self._add_task(reloaded)
        self.assertIs(
            reloaded.get_task(reloaded_index).status,
            TaskStatus.COMPLETED,
        )

        config.set(config.inpaintMode, InpaintMode.STTN_AUTO)
        reloaded.resync_completion_states()
        self.assertIs(reloaded.get_task(reloaded_index).status, TaskStatus.PENDING)

        config.set(config.inpaintMode, InpaintMode.SUBTITLE_FIXED_WATERMARK)
        reloaded.resync_completion_states()
        self.assertIs(reloaded.get_task(reloaded_index).status, TaskStatus.COMPLETED)

        component.close()
        reloaded.close()


if __name__ == "__main__":
    unittest.main()
