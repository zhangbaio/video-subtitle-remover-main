import io
import unittest
from unittest import mock

from backend.tools import progress


class SafeProgressTests(unittest.TestCase):
    def test_progress_and_write_work_without_console_streams(self):
        with (
            mock.patch.object(progress.sys, "stdout", None),
            mock.patch.object(progress.sys, "stderr", None),
            mock.patch.object(progress.sys, "__stdout__", None),
            mock.patch.object(progress.sys, "__stderr__", None),
        ):
            bar = progress.safe_tqdm(total=3, desc="windowed")
            try:
                bar.update(2)
                bar.write("still running")
                progress.safe_tqdm_write("worker message")
                self.assertEqual(bar.n, 2)
            finally:
                bar.close()

    def test_preferred_writable_stream_is_preserved(self):
        stream = io.StringIO()
        self.assertIs(progress.safe_console_stream(stream), stream)


if __name__ == "__main__":
    unittest.main()
