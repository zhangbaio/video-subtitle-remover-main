import unittest
from unittest import mock

from backend.tools import subprocess_utils


class FakeStartupInfo:
    def __init__(self):
        self.dwFlags = 0
        self.wShowWindow = None


class HiddenSubprocessKwargsTests(unittest.TestCase):
    def test_non_windows_returns_empty_options(self):
        with mock.patch.object(subprocess_utils.os, "name", "posix"):
            self.assertEqual(subprocess_utils.hidden_subprocess_kwargs(), {})

    def test_windows_sets_no_console_and_hidden_window_flags(self):
        with (
            mock.patch.object(subprocess_utils.os, "name", "nt"),
            mock.patch.object(
                subprocess_utils.subprocess,
                "STARTUPINFO",
                FakeStartupInfo,
                create=True,
            ),
            mock.patch.object(
                subprocess_utils.subprocess,
                "STARTF_USESHOWWINDOW",
                0x01,
                create=True,
            ),
            mock.patch.object(
                subprocess_utils.subprocess,
                "SW_HIDE",
                0,
                create=True,
            ),
            mock.patch.object(
                subprocess_utils.subprocess,
                "CREATE_NO_WINDOW",
                0x08000000,
                create=True,
            ),
        ):
            kwargs = subprocess_utils.hidden_subprocess_kwargs()

        self.assertEqual(kwargs["creationflags"], 0x08000000)
        self.assertTrue(kwargs["startupinfo"].dwFlags & 0x01)
        self.assertEqual(kwargs["startupinfo"].wShowWindow, 0)


if __name__ == "__main__":
    unittest.main()
