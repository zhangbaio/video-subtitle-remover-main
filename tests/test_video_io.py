import threading
import time
import unittest
from types import SimpleNamespace
from unittest import mock

import numpy as np

from backend.tools import video_io


class _FakeStdin:
    def __init__(self, error=None):
        self.error = error
        self.closed = False
        self.failed = threading.Event()
        self.writes = []

    def write(self, data):
        if self.error is not None:
            self.failed.set()
            raise self.error
        payload = bytes(data)
        self.writes.append(payload)
        return len(payload)

    def close(self):
        self.closed = True


class _BlockingStdin(_FakeStdin):
    def __init__(self):
        super().__init__()
        self.first_write_started = threading.Event()
        self.allow_first_write = threading.Event()
        self.process_stopped = threading.Event()

    def stop_process(self):
        self.process_stopped.set()
        self.allow_first_write.set()

    def write(self, data):
        if not self.first_write_started.is_set():
            self.first_write_started.set()
            if not self.allow_first_write.wait(timeout=5):
                raise TimeoutError('test did not unblock the FFmpeg pipe')
        if self.process_stopped.is_set():
            raise BrokenPipeError('fake FFmpeg process stopped')
        return super().write(data)


class _BlockingCloseStdin(_FakeStdin):
    def __init__(self):
        super().__init__()
        self.close_started = threading.Event()
        self.allow_close = threading.Event()

    def stop_process(self):
        self.allow_close.set()

    def close(self):
        self.close_started.set()
        if not self.allow_close.wait(timeout=5):
            raise TimeoutError('test did not unblock stdin.close()')
        super().close()


class _FakeProcess:
    def __init__(self, stdin, returncode=0, terminate_requires_kill=False):
        self.stdin = stdin
        self.returncode = returncode
        self.terminate_requires_kill = terminate_requires_kill
        self.wait_calls = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def wait(self, timeout=None):
        self.wait_calls.append(timeout)
        if (
            self.terminate_requires_kill
            and self.terminate_calls
            and not self.kill_calls
        ):
            raise video_io.subprocess.TimeoutExpired('ffmpeg', timeout)
        return self.returncode

    def terminate(self):
        self.terminate_calls += 1
        if not self.terminate_requires_kill and hasattr(
            self.stdin,
            'stop_process',
        ):
            self.stdin.stop_process()

    def kill(self):
        self.kill_calls += 1
        if hasattr(self.stdin, 'stop_process'):
            self.stdin.stop_process()


class FFmpegVideoWriterTests(unittest.TestCase):
    def _create_writer(
        self,
        stdin=None,
        queue_size=16,
        hidden_kwargs=None,
        returncode=0,
        terminate_requires_kill=False,
        process_manager=None,
        bitrate_kbps=4500,
    ):
        stdin = stdin or _FakeStdin()
        process = _FakeProcess(
            stdin,
            returncode=returncode,
            terminate_requires_kill=terminate_requires_kill,
        )
        if hidden_kwargs is None:
            hidden_kwargs = {'creationflags': 1234}
        popen = mock.patch.object(
            video_io.subprocess,
            'Popen',
            return_value=process,
        )
        ffmpeg = mock.patch.object(
            video_io.FFmpegCLI,
            'instance',
            return_value=SimpleNamespace(ffmpeg_path='ffmpeg'),
        )
        hidden = mock.patch.object(
            video_io,
            'hidden_subprocess_kwargs',
            return_value=hidden_kwargs,
        )
        if process_manager is None:
            process_manager = mock.Mock()
            process_manager.add_process.return_value = 'ffmpeg-writer-test'
        manager = mock.patch.object(
            video_io.ProcessManager,
            'instance',
            return_value=process_manager,
        )
        popen_mock = popen.start()
        ffmpeg.start()
        hidden.start()
        manager.start()
        self.addCleanup(popen.stop)
        self.addCleanup(ffmpeg.stop)
        self.addCleanup(hidden.stop)
        self.addCleanup(manager.stop)
        self._last_process = process
        self._last_process_manager = process_manager
        writer = video_io.FFmpegVideoWriter(
            'out.mp4',
            30,
            (2, 2),
            queue_size=queue_size,
            bitrate_kbps=bitrate_kbps,
        )
        return writer, process, popen_mock

    @staticmethod
    def _frame(value):
        return np.full((2, 2, 3), value, dtype=np.uint8)

    def test_configured_bitrate_is_passed_to_ffmpeg(self):
        writer, _, popen = self._create_writer(bitrate_kbps=6789)
        self.addCleanup(writer.release)

        command = popen.call_args.args[0]
        self.assertEqual(command[command.index('-b:v') + 1], '6789k')
        self.assertEqual(command[command.index('-minrate') + 1], '6789k')
        self.assertEqual(command[command.index('-maxrate') + 1], '6789k')
        self.assertEqual(command[command.index('-bufsize') + 1], '13578k')
        self.assertNotIn('-crf', command)
        self.assertEqual(writer.bitrate_kbps, 6789)

    def test_invalid_bitrate_is_rejected_before_starting_ffmpeg(self):
        with self.assertRaisesRegex(ValueError, 'bitrate_kbps'):
            video_io.FFmpegVideoWriter('out.mp4', 30, (2, 2), bitrate_kbps=0)

    def test_writes_frames_in_order_and_release_waits_for_all_frames(self):
        stdin = _BlockingStdin()
        writer, process, _ = self._create_writer(stdin=stdin, queue_size=2)

        first = self._frame(1)
        writer.write(first)
        self.assertTrue(stdin.first_write_started.wait(timeout=2))
        # Mutating the caller's array must not mutate the queued payload.
        first.fill(99)
        writer.write(self._frame(2))

        released = threading.Event()

        def release():
            writer.release()
            released.set()

        release_thread = threading.Thread(target=release)
        release_thread.start()
        self.assertFalse(released.wait(timeout=0.1))

        stdin.allow_first_write.set()
        release_thread.join(timeout=2)

        self.assertFalse(release_thread.is_alive())
        self.assertTrue(released.is_set())
        self.assertEqual(
            stdin.writes,
            [self._frame(1).tobytes(), self._frame(2).tobytes()],
        )
        self.assertTrue(stdin.closed)
        self.assertEqual(len(process.wait_calls), 1)
        self.assertGreater(process.wait_calls[0], 599)
        self.assertLessEqual(process.wait_calls[0], 600)

    def test_registers_process_until_successful_release(self):
        writer, process, _ = self._create_writer()
        manager = self._last_process_manager

        manager.add_process.assert_called_once_with(
            process,
        )
        manager.remove_process.assert_not_called()

        writer.release()
        writer.release()

        manager.remove_process.assert_called_once_with('ffmpeg-writer-test')

    def test_registration_rejected_during_shutdown_stops_process_and_raises(self):
        manager = mock.Mock()
        manager.add_process.return_value = None

        with self.assertRaisesRegex(RuntimeError, 'shutdown is in progress'):
            self._create_writer(process_manager=manager)

        process = self._last_process
        manager.add_process.assert_called_once_with(
            process,
        )
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 0)
        self.assertEqual(len(process.wait_calls), 1)
        manager.remove_process.assert_not_called()

    def test_constructor_failure_after_registration_stops_and_unregisters(self):
        manager = mock.Mock()
        manager.add_process.return_value = 'registered-before-thread-start'

        with (
            mock.patch.object(
                video_io.threading.Thread,
                'start',
                side_effect=RuntimeError('thread start failed'),
            ),
            self.assertRaisesRegex(RuntimeError, 'thread start failed'),
        ):
            self._create_writer(process_manager=manager)

        process = self._last_process
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(len(process.wait_calls), 1)
        manager.remove_process.assert_called_once_with(
            'registered-before-thread-start'
        )

    def test_live_process_keeps_registration_for_shutdown_retry(self):
        writer, process, _ = self._create_writer()
        manager = self._last_process_manager
        process.poll = mock.Mock(return_value=None)

        self.assertFalse(writer._unregister_process())
        manager.remove_process.assert_not_called()

        process.poll.return_value = 0
        self.assertTrue(writer._unregister_process())
        manager.remove_process.assert_called_once_with('ffmpeg-writer-test')

    def test_queue_is_bounded_and_applies_backpressure(self):
        stdin = _BlockingStdin()
        writer, _, _ = self._create_writer(stdin=stdin, queue_size=1)
        self.addCleanup(writer.release)
        self.addCleanup(stdin.allow_first_write.set)

        writer.write(self._frame(1))
        self.assertTrue(stdin.first_write_started.wait(timeout=2))
        writer.write(self._frame(2))

        third_write_done = threading.Event()

        def write_third():
            writer.write(self._frame(3))
            third_write_done.set()

        write_thread = threading.Thread(target=write_third)
        write_thread.start()
        self.assertFalse(third_write_done.wait(timeout=0.1))
        self.assertEqual(writer._frame_queue.maxsize, 1)

        stdin.allow_first_write.set()
        write_thread.join(timeout=2)
        self.assertFalse(write_thread.is_alive())
        self.assertTrue(third_write_done.is_set())

    def test_full_queue_write_times_out_and_force_stops_ffmpeg(self):
        stdin = _BlockingStdin()
        writer, process, _ = self._create_writer(
            stdin=stdin,
            queue_size=1,
            terminate_requires_kill=True,
        )

        writer.write(self._frame(1))
        self.assertTrue(stdin.first_write_started.wait(timeout=2))
        writer.write(self._frame(2))

        started = time.monotonic()
        with (
            mock.patch.object(writer, '_WRITE_TIMEOUT_SECONDS', 0.05),
            mock.patch.object(writer, '_FORCE_STOP_TIMEOUT_SECONDS', 0.05),
            self.assertRaisesRegex(TimeoutError, 'accept a video frame'),
        ):
            writer.write(self._frame(3))
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertFalse(writer._writer_thread.is_alive())
        self._last_process_manager.remove_process.assert_called_once_with(
            'ffmpeg-writer-test'
        )
        with self.assertRaisesRegex(TimeoutError, 'accept a video frame'):
            writer.release()
        self._last_process_manager.remove_process.assert_called_once_with(
            'ffmpeg-writer-test'
        )

    def test_release_timeout_force_stops_blocked_writer(self):
        stdin = _BlockingStdin()
        writer, process, _ = self._create_writer(
            stdin=stdin,
            queue_size=1,
            terminate_requires_kill=True,
        )

        writer.write(self._frame(1))
        self.assertTrue(stdin.first_write_started.wait(timeout=2))

        started = time.monotonic()
        with (
            mock.patch.object(writer, '_FINALIZE_TIMEOUT_SECONDS', 0.05),
            mock.patch.object(writer, '_FORCE_STOP_TIMEOUT_SECONDS', 0.05),
            self.assertRaisesRegex(TimeoutError, 'finalizing'),
        ):
            writer.release()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)
        self.assertFalse(writer._writer_thread.is_alive())
        self.assertTrue(stdin.closed)
        with self.assertRaisesRegex(TimeoutError, 'finalizing'):
            writer.release()

    def test_release_deadline_includes_waiting_for_active_producer(self):
        stdin = _BlockingStdin()
        writer, process, _ = self._create_writer(
            stdin=stdin,
            queue_size=1,
            terminate_requires_kill=True,
        )

        writer.write(self._frame(1))
        self.assertTrue(stdin.first_write_started.wait(timeout=2))
        writer.write(self._frame(2))

        producer_errors = []
        producer_started = threading.Event()

        def blocked_write():
            producer_started.set()
            try:
                writer.write(self._frame(3))
            except BaseException as error:
                producer_errors.append(error)

        producer = threading.Thread(target=blocked_write)
        producer.start()
        self.assertTrue(producer_started.wait(timeout=2))
        time.sleep(0.02)

        started = time.monotonic()
        with (
            mock.patch.object(writer, '_FINALIZE_TIMEOUT_SECONDS', 0.05),
            mock.patch.object(writer, '_FORCE_STOP_TIMEOUT_SECONDS', 0.05),
            self.assertRaisesRegex(TimeoutError, 'finalizing'),
        ):
            writer.release()
        elapsed = time.monotonic() - started
        producer.join(timeout=1)

        self.assertLess(elapsed, 0.5)
        self.assertFalse(producer.is_alive())
        self.assertTrue(producer_errors)
        self.assertIsInstance(producer_errors[0], TimeoutError)
        self.assertGreaterEqual(process.terminate_calls, 1)
        self.assertGreaterEqual(process.kill_calls, 1)

    def test_release_deadline_also_bounds_blocked_stdin_close(self):
        stdin = _BlockingCloseStdin()
        writer, process, _ = self._create_writer(
            stdin=stdin,
            terminate_requires_kill=True,
        )

        started = time.monotonic()
        with (
            mock.patch.object(writer, '_FINALIZE_TIMEOUT_SECONDS', 0.05),
            mock.patch.object(writer, '_FORCE_STOP_TIMEOUT_SECONDS', 0.05),
            self.assertRaisesRegex(TimeoutError, 'finalizing'),
        ):
            writer.release()
        elapsed = time.monotonic() - started

        self.assertLess(elapsed, 0.5)
        self.assertTrue(stdin.close_started.is_set())
        self.assertTrue(stdin.closed)
        self.assertEqual(process.terminate_calls, 1)
        self.assertEqual(process.kill_calls, 1)

    def test_release_propagates_background_broken_pipe(self):
        error = BrokenPipeError('ffmpeg pipe closed')
        stdin = _FakeStdin(error=error)
        writer, process, _ = self._create_writer(stdin=stdin)

        writer.write(self._frame(1))
        self.assertTrue(stdin.failed.wait(timeout=2))

        with self.assertRaisesRegex(BrokenPipeError, 'ffmpeg pipe closed'):
            writer.release()
        # Idempotent cleanup must not make an earlier encoder failure appear
        # successful to a caller that releases the writer again.
        with self.assertRaisesRegex(BrokenPipeError, 'ffmpeg pipe closed'):
            writer.release()

        self.assertTrue(stdin.closed)
        self.assertEqual(len(process.wait_calls), 1)
        self.assertGreater(process.wait_calls[0], 599)
        self.assertLessEqual(process.wait_calls[0], 600)

    def test_release_propagates_other_background_exception(self):
        error = RuntimeError('encoder write failed')
        stdin = _FakeStdin(error=error)
        writer, _, _ = self._create_writer(stdin=stdin)

        writer.write(self._frame(1))
        self.assertTrue(stdin.failed.wait(timeout=2))

        with self.assertRaisesRegex(RuntimeError, 'encoder write failed'):
            writer.release()

    def test_release_reports_nonzero_ffmpeg_exit_code(self):
        writer, _, _ = self._create_writer(returncode=7)
        manager = self._last_process_manager

        with self.assertRaisesRegex(RuntimeError, 'exited with code 7'):
            writer.release()
        with self.assertRaisesRegex(RuntimeError, 'exited with code 7'):
            writer.release()
        manager.remove_process.assert_called_once_with('ffmpeg-writer-test')

    def test_write_after_release_fails(self):
        writer, _, _ = self._create_writer()
        writer.release()

        with self.assertRaisesRegex(RuntimeError, 'released'):
            writer.write(self._frame(1))

    def test_ffmpeg_process_keeps_hidden_window_options(self):
        hidden_kwargs = {
            'creationflags': 0x08000000,
            'startupinfo': object(),
        }
        writer, _, popen = self._create_writer(hidden_kwargs=hidden_kwargs)
        writer.release()

        kwargs = popen.call_args.kwargs
        self.assertEqual(kwargs['creationflags'], 0x08000000)
        self.assertIs(kwargs['startupinfo'], hidden_kwargs['startupinfo'])
        self.assertIs(kwargs['stdin'], video_io.subprocess.PIPE)
        self.assertIs(kwargs['stdout'], video_io.subprocess.DEVNULL)
        self.assertIs(kwargs['stderr'], video_io.subprocess.DEVNULL)

    def test_default_queue_is_bounded_to_sixteen_frames(self):
        writer, _, _ = self._create_writer()
        self.addCleanup(writer.release)

        self.assertEqual(writer._frame_queue.maxsize, 16)

    def test_queue_size_must_be_positive(self):
        with self.assertRaisesRegex(ValueError, 'queue_size'):
            self._create_writer(queue_size=0)


if __name__ == '__main__':
    unittest.main()
