import os
from pathlib import Path
import sys
import unittest


os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
WORKSPACE_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKSPACE_ROOT))

try:
    from PySide6.QtWidgets import QApplication

    from backend.config import config
    from backend.tools.constant import InpaintMode
    from ui.component.task_list_component import TaskOptions
    from ui.home_interface import HomeInterface

    GUI_AVAILABLE = True
except ImportError:
    GUI_AVAILABLE = False


@unittest.skipUnless(GUI_AVAILABLE, "PySide6/qfluentwidgets is not installed")
class CombinedModeUiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self.original_mode = config.inpaintMode.value
        self.home = HomeInterface()
        self.task_index = self.home.task_list_component.add_task(
            r"D:\combined-ui-test.mp4",
            batch_id="combined-ui-test",
        )
        self.home.task_list_component.set_current_task(self.task_index)
        self.home.video_path = r"D:\combined-ui-test.mp4"
        self.home.frame_width = 1000
        self.home.frame_height = 1000
        self.home.frame_count = 100
        self.home._displayed_frame_no = 7
        self.home._get_video_dimensions = lambda _path: (1000, 1000)
        self.home.video_display_component.set_video_parameters(
            1000, 1000, 1, 1
        )

    def tearDown(self):
        config.set(config.inpaintMode, self.original_mode)
        self.home.close()
        self.app.processEvents()

    def _set_mode(self, mode):
        config.set(config.inpaintMode, mode)
        self.home.on_inpaint_mode_changed(mode)

    def test_combined_fixed_layers_keep_subtitle_and_watermark_separate(self):
        self._set_mode(InpaintMode.SUBTITLE_FIXED_WATERMARK)
        subtitle_areas = [(0.70, 0.80, 0.20, 0.80)]
        self.home.video_display_component.set_selection_rects(subtitle_areas)
        self.home.selections_changed(subtitle_areas)

        self.home.toggle_selection_target()
        fixed_watermark_preview = [(0.10, 0.20, 0.10, 0.20)]
        self.home.video_display_component.set_selection_rects(
            fixed_watermark_preview
        )
        self.home.selections_changed(fixed_watermark_preview)

        task_list = self.home.task_list_component
        self.assertEqual(
            task_list.get_task_option(self.task_index, TaskOptions.SUB_AREAS),
            subtitle_areas,
        )
        self.assertTrue(
            task_list.get_task_option(
                self.task_index, TaskOptions.FIXED_WATERMARK_AREAS
            )
        )
        self.assertEqual(
            self.home.video_display_component.overlay_selection_rects,
            subtitle_areas,
        )

    def test_combined_moving_template_is_unique_and_old_modes_reset_overlay(self):
        self._set_mode(InpaintMode.SUBTITLE_MOVING_WATERMARK)
        self.home.toggle_selection_target()
        attempted_templates = [
            (0.10, 0.20, 0.10, 0.20),
            (0.20, 0.30, 0.20, 0.30),
        ]
        self.home.video_display_component.set_selection_rects(attempted_templates)
        self.home.selections_changed(attempted_templates)

        self.assertEqual(
            len(self.home.video_display_component.get_selection_rects()), 1
        )
        self.assertEqual(
            self.home.task_list_component.get_task_option(
                self.task_index,
                TaskOptions.MOVING_WATERMARK_REFERENCE_FRAME_NO,
            ),
            7,
        )

        self._set_mode(InpaintMode.STTN_AUTO)
        self.assertFalse(self.home.selection_target_button.isVisible())
        self.assertFalse(
            self.home.video_display_component.overlay_selection_rects
        )
        self.assertTrue(self.home.auto_area_button.isEnabled())

    def test_switching_to_moving_watermark_loads_saved_reference_without_overwrite(self):
        self._set_mode(InpaintMode.SUBTITLE_MOVING_WATERMARK)
        source_path = r"D:\moving-template-source.mp4"
        template_area = (0.10, 0.20, 0.10, 0.20)
        task_list = self.home.task_list_component
        task_list.update_task_option(
            self.task_index,
            TaskOptions.MOVING_WATERMARK_TEMPLATE_AREA,
            template_area,
        )
        task_list.update_task_option(
            self.task_index,
            TaskOptions.MOVING_WATERMARK_REFERENCE_FRAME_NO,
            42,
        )
        task_list.update_task_option(
            self.task_index,
            TaskOptions.MOVING_WATERMARK_TEMPLATE_SOURCE_PATH,
            source_path,
        )
        loaded_paths = []
        seeked_frames = []

        def load_video(path):
            loaded_paths.append(path)
            self.home.video_path = path
            return True

        self.home.load_video = load_video
        self.home._seek_preview_to_frame = (
            lambda frame_no: seeked_frames.append(frame_no) or True
        )
        self.home.toggle_selection_target()

        self.assertEqual(loaded_paths[-1], source_path)
        self.assertEqual(seeked_frames, [42])
        self.assertEqual(
            task_list.get_task_option(
                self.task_index, TaskOptions.MOVING_WATERMARK_TEMPLATE_AREA
            ),
            template_area,
        )

        self.home.toggle_selection_target()
        self.assertEqual(
            loaded_paths[-1], task_list.get_task(self.task_index).path
        )

    def test_manual_auto_area_result_is_bound_to_request_context(self):
        self._set_mode(InpaintMode.SUBTITLE_FIXED_WATERMARK)
        first_path = self.home.task_list_component.get_task(self.task_index).path
        self.home.video_path = first_path
        self.home._auto_area_request_token = 10
        self.home._active_auto_area_request = (
            10,
            self.task_index,
            self.home._auto_area_request_path(first_path),
        )
        self.home.set_auto_area_button_running(True, 10)

        second_index = self.home.task_list_component.add_task(
            r"D:\combined-ui-test-2.mp4",
            batch_id="combined-ui-test",
        )
        self.home.task_list_component.set_current_task(second_index)
        self.home.video_path = r"D:\combined-ui-test-2.mp4"
        self.home.on_auto_subtitle_area_detected(
            self.task_index,
            first_path,
            10,
            [(700, 800, 200, 800)],
            0.9,
        )

        self.assertNotEqual(
            self.home.task_list_component.get_task_option(
                second_index, TaskOptions.SUB_AREAS, []
            ),
            [(0.7, 0.8, 0.2, 0.8)],
        )
        self.home.set_auto_area_button_running(False, 10)

        self.home.task_list_component.set_current_task(self.task_index)
        self.home.video_path = first_path
        self.home._active_auto_area_request = (
            11,
            self.task_index,
            self.home._auto_area_request_path(first_path),
        )
        self.home.set_auto_area_button_running(True, 11)
        self.home.on_auto_subtitle_area_detected(
            self.task_index,
            first_path,
            10,
            [(100, 200, 100, 200)],
            0.5,
        )
        self.home.on_auto_subtitle_area_detected(
            self.task_index,
            first_path,
            11,
            [(700, 800, 200, 800)],
            0.9,
        )
        self.assertEqual(
            self.home.task_list_component.get_task_option(
                self.task_index, TaskOptions.SUB_AREAS, []
            ),
            [(0.7, 0.8, 0.2, 0.8)],
        )
        self.home.set_auto_area_button_running(False, 11)

    def test_auto_area_and_processing_lock_all_batch_controls(self):
        self._set_mode(InpaintMode.SUBTITLE_MOVING_WATERMARK)
        path = self.home.task_list_component.get_task(self.task_index).path
        self.home._active_auto_area_request = (
            3,
            self.task_index,
            self.home._auto_area_request_path(path),
        )
        self.home.set_auto_area_button_running(True, 3)
        self.assertFalse(self.home.run_button.isEnabled())
        self.assertFalse(self.home.task_list_component.isEnabled())
        self.assertFalse(
            self.home.setting_interface.subtitle_detect_model_combo.isEnabled()
        )
        self.assertFalse(
            self.home.setting_interface.auto_subtitle_area_selection.isEnabled()
        )
        self.assertFalse(
            self.home.setting_interface.hardware_acceleration.isEnabled()
        )
        self.assertFalse(
            self.home.setting_interface.moving_watermark_fast_mode.isEnabled()
        )

        self.home.set_auto_area_button_running(False, 3)
        self.home._toggle_buttons(False)
        self.assertFalse(
            self.home.setting_interface.inpaint_mode_combo.comboBox.isEnabled()
        )
        self.assertFalse(
            self.home.setting_interface.subtitle_detect_model_combo.isEnabled()
        )
        self.assertFalse(
            self.home.setting_interface.auto_subtitle_area_selection.isEnabled()
        )
        self.assertFalse(
            self.home.setting_interface.hardware_acceleration.isEnabled()
        )
        self.assertFalse(
            self.home.setting_interface.moving_watermark_fast_mode.isEnabled()
        )

        self.home._toggle_buttons(True)
        self.assertTrue(
            self.home.setting_interface.inpaint_mode_combo.comboBox.isEnabled()
        )
        self.assertTrue(
            self.home.setting_interface.subtitle_detect_model_combo.isEnabled()
        )
        self.assertTrue(
            self.home.setting_interface.auto_subtitle_area_selection.isEnabled()
        )
        self.assertTrue(
            self.home.setting_interface.hardware_acceleration.isEnabled()
        )
        self.assertTrue(
            self.home.setting_interface.moving_watermark_fast_mode.isEnabled()
        )

        self._set_mode(InpaintMode.FIXED_WATERMARK)
        self.assertFalse(
            self.home.setting_interface.subtitle_detect_model_combo.isEnabled()
        )
        self.assertFalse(
            self.home.setting_interface.auto_subtitle_area_selection.isEnabled()
        )
        self.assertFalse(
            self.home.setting_interface.moving_watermark_fast_mode.isEnabled()
        )
        self.assertTrue(
            self.home.setting_interface.inpaint_mode_combo.comboBox.isEnabled()
        )


if __name__ == "__main__":
    unittest.main()
