import os
import threading
import types
import unittest
from concurrent.futures import Future
from unittest import mock

from backend.tools import process_manager
from backend.tools.process_manager import ProcessManager
from ui import home_interface
from ui.home_interface import (
    HomeInterface,
    _DaemonSingleWorkerExecutor,
    _exit_after_parent_process,
    _start_parent_death_watchdog,
)


class _FakeProcess:
    def __init__(self, pid=45678, alive=False):
        self.pid = pid
        self.alive = alive
        self.started = 0
        self.join_calls = []
        self.terminate_calls = 0
        self.kill_calls = 0

    def start(self):
        self.started += 1
        self.alive = True

    def is_alive(self):
        return self.alive

    def join(self, timeout=None):
        self.join_calls.append(timeout)

    def terminate(self):
        self.terminate_calls += 1
        self.alive = False

    def kill(self):
        self.kill_calls += 1
        self.alive = False


class _FakeQueue:
    def __init__(self):
        self.items = []
        self.closed = False
        self.cancelled_join = False

    def put_nowait(self, item):
        self.items.append(item)

    def close(self):
        self.closed = True

    def cancel_join_thread(self):
        self.cancelled_join = True


class _FakeRemoteCaller:
    def __init__(self):
        self.queue = object()
        self.stopped = False

    def stop(self):
        self.stopped = True


class _FakeManager:
    def __init__(self):
        self.add_calls = []
        self.terminated = []
        self.removed = []
        self.shutdown_calls = 0

    def add_process(self, process):
        self.add_calls.append(process)
        return "worker-id"

    def terminate_by_process(self, process):
        self.terminated.append(process)
        process.alive = False

    def remove_process(self, process_id):
        self.removed.append(process_id)
        return True

    def shutdown(self):
        self.shutdown_calls += 1


class _FailingRegistrationManager(_FakeManager):
    def add_process(self, process):
        self.add_calls.append(process)
        raise RuntimeError("registration failed")


class _RejectingManager(_FakeManager):
    def add_process(self, process):
        self.add_calls.append(process)
        self.terminate_by_process(process)
        return None


class _LifecycleHarness:
    _ensure_subtitle_worker = HomeInterface._ensure_subtitle_worker
    _dispose_subtitle_worker = HomeInterface._dispose_subtitle_worker
    shutdown_workers = HomeInterface.shutdown_workers

    def __init__(self):
        self._worker_lifecycle_lock = threading.RLock()
        self._closing = False
        self._executors_shutdown = False
        self._stop_event = threading.Event()
        self._worker_thread = None
        self.worker_process = None
        self.worker_process_id = None
        self.worker_command_queue = None
        self.worker_remote_caller = None
        self.running_process = None
        self.last_worker_job_succeeded = False
        self.auto_area_executor = mock.Mock()
        self.subtitle_timeline_executor = mock.Mock()
        self._cancel_subtitle_timeline_prefetch = mock.Mock(return_value=[])
        self._cancel_auto_area_detections = mock.Mock(return_value=[])
        self._resume_auto_area_preprocessing = mock.Mock()
        self._register_worker_callbacks = mock.Mock()
        self._video_cap_lock = threading.Lock()
        self.video_cap = None


class ProcessManagerTests(unittest.TestCase):
    def setUp(self):
        with mock.patch.object(process_manager.atexit, "register"):
            self.manager = ProcessManager()

    def test_shutdown_rejects_and_cleans_late_process(self):
        process = _FakeProcess(alive=True)
        self.manager.begin_shutdown()
        with mock.patch.object(self.manager, "terminate_by_process") as terminate:
            self.assertIsNone(self.manager.add_process(process))
        terminate.assert_called_once_with(process)
        self.assertEqual(self.manager.processes, {})

    def test_shutdown_uses_synchronous_cleanup_without_thread_pool(self):
        process = _FakeProcess(alive=True)
        self.manager.processes = {
            "worker": process,
            "pid": 45679,
        }

        with (
            mock.patch.object(self.manager, "terminate_by_process") as terminate_process,
            mock.patch.object(self.manager, "terminate_by_pid") as terminate_pid,
            mock.patch.object(
                process_manager.concurrent.futures,
                "ThreadPoolExecutor",
            ) as executor,
        ):
            self.manager.shutdown()

        executor.assert_not_called()
        terminate_process.assert_called_once_with(process)
        terminate_pid.assert_called_once_with(45679)
        self.assertFalse(self.manager.accepting_processes)
        self.assertEqual(self.manager.processes, {})

    def test_parallel_cleanup_falls_back_when_executor_submit_is_rejected(self):
        first = _FakeProcess(pid=45678, alive=True)
        second = _FakeProcess(pid=45679, alive=True)
        self.manager.processes = {"first": first, "second": second}
        executor = mock.Mock()
        executor.submit.side_effect = RuntimeError(
            "cannot schedule new futures after interpreter shutdown"
        )

        with (
            mock.patch.object(
                process_manager.concurrent.futures,
                "ThreadPoolExecutor",
                return_value=executor,
            ),
            mock.patch.object(
                self.manager,
                "terminate_by_process",
                return_value=True,
            ) as terminate,
        ):
            self.manager.terminate_all()

        self.assertEqual(terminate.call_args_list, [mock.call(first), mock.call(second)])
        executor.shutdown.assert_called_once_with(wait=True)
        self.assertEqual(self.manager.processes, {})

    def test_process_tree_is_terminated_before_parent_fallback(self):
        process = _FakeProcess(alive=True)

        def terminate_tree(_pid):
            process.alive = False
            return True

        with mock.patch.object(
            self.manager, "terminate_by_pid", side_effect=terminate_tree
        ) as terminate_tree_mock:
            self.manager.terminate_by_process(process)

        terminate_tree_mock.assert_called_once_with(process.pid)
        self.assertEqual(process.terminate_calls, 0)
        self.assertEqual(process.kill_calls, 0)

    def test_windows_tree_kill_uses_taskkill_t_flag(self):
        with (
            mock.patch.object(process_manager.platform, "system", return_value="Windows"),
            mock.patch.object(process_manager.subprocess, "run") as run,
            mock.patch.object(
                process_manager, "hidden_subprocess_kwargs", return_value={"creationflags": 8}
            ),
        ):
            run.return_value.returncode = 0
            self.manager.terminate_by_pid(45678)

        command = run.call_args.args[0]
        self.assertEqual(command[:4], ["taskkill", "/F", "/T", "/PID"])
        self.assertEqual(command[4], "45678")
        self.assertEqual(run.call_args.kwargs["creationflags"], 8)

    def test_windows_taskkill_nonzero_is_retried_and_reported(self):
        result = types.SimpleNamespace(returncode=5)
        with (
            mock.patch.object(process_manager.platform, "system", return_value="Windows"),
            mock.patch.object(process_manager.subprocess, "run", return_value=result) as run,
        ):
            self.assertFalse(self.manager.terminate_by_pid(45678))
        self.assertEqual(run.call_count, 2)

    def test_windows_taskkill_timeout_is_retried(self):
        success = types.SimpleNamespace(returncode=0)
        with (
            mock.patch.object(process_manager.platform, "system", return_value="Windows"),
            mock.patch.object(
                process_manager.subprocess,
                "run",
                side_effect=[process_manager.subprocess.TimeoutExpired("taskkill", 5), success],
            ) as run,
        ):
            self.assertTrue(self.manager.terminate_by_pid(45678))
        self.assertEqual(run.call_count, 2)

    def test_failed_tree_kill_falls_back_to_parent_terminate(self):
        process = _FakeProcess(alive=True)
        with mock.patch.object(self.manager, "terminate_by_pid", return_value=False):
            self.assertTrue(self.manager.terminate_by_process(process))
        self.assertEqual(process.terminate_calls, 1)

    def test_posix_tree_is_ordered_deepest_first(self):
        result = types.SimpleNamespace(stdout="100 1\n101 100\n102 101\n103 100\n")
        with mock.patch.object(process_manager.subprocess, "run", return_value=result):
            self.assertEqual(
                self.manager._posix_process_tree(100),
                [102, 101, 103, 100],
            )

    def test_invalid_or_current_pid_is_never_killed(self):
        with mock.patch.object(process_manager.subprocess, "run") as run:
            self.manager.terminate_by_pid(0)
            self.manager.terminate_by_pid(os.getpid())
        run.assert_not_called()


class HomeWorkerLifecycleTests(unittest.TestCase):
    def test_optional_prescan_executor_uses_daemon_worker_and_nonblocking_shutdown(self):
        executor = _DaemonSingleWorkerExecutor("test-prescan-daemon")
        release = threading.Event()
        started = threading.Event()

        def blocked_native_call():
            started.set()
            release.wait(timeout=2)

        future = executor.submit(blocked_native_call)
        self.assertTrue(started.wait(timeout=1))
        self.assertTrue(executor._thread.daemon)

        shutdown_thread = threading.Thread(
            target=lambda: executor.shutdown(wait=False, cancel_futures=True)
        )
        shutdown_thread.start()
        shutdown_thread.join(timeout=0.5)
        self.assertFalse(shutdown_thread.is_alive())
        self.assertFalse(future.done())

        release.set()
        executor._thread.join(timeout=1)
        self.assertFalse(executor._thread.is_alive())

    def test_pause_auto_area_preprocessing_stops_without_waiting_for_native_job(self):
        home = types.SimpleNamespace(
            _auto_area_condition=threading.Condition(),
            _auto_area_preprocessing_paused=False,
            _auto_area_active_jobs=1,
            _stop_event=threading.Event(),
            _closing=False,
        )
        home._stop_event.set()

        with self.assertRaises(home_interface.SubtitleDetectionCancelled):
            HomeInterface._pause_auto_area_preprocessing(home)

    def test_unhealthy_auto_area_does_not_block_removal_start(self):
        home = types.SimpleNamespace(
            _auto_area_condition=threading.Condition(),
            _auto_area_preprocessing_paused=False,
            _auto_area_active_jobs=1,
            _auto_area_unhealthy=True,
            _stop_event=threading.Event(),
            _closing=False,
        )

        HomeInterface._pause_auto_area_preprocessing(home)

        self.assertTrue(home._auto_area_preprocessing_paused)

    def test_auto_area_timeout_disables_prescan_instead_of_waiting_forever(self):
        future = Future()
        log_signal = mock.Mock()
        home = types.SimpleNamespace(
            AUTO_AREA_WAIT_TIMEOUT_SECONDS=0.0,
            _auto_area_lock=threading.Lock(),
            _auto_area_condition=threading.Condition(),
            _auto_area_unhealthy=False,
            _auto_area_cancel_event=threading.Event(),
            _auto_area_futures={"episode": future},
            _manual_auto_area_future=None,
            _stop_event=threading.Event(),
            _closing=False,
            append_log_signal=log_signal,
        )
        home._disable_auto_area_preprocessing = types.MethodType(
            HomeInterface._disable_auto_area_preprocessing,
            home,
        )

        with self.assertRaisesRegex(TimeoutError, "timed out"):
            HomeInterface._wait_for_auto_area_future(home, future)

        self.assertTrue(home._auto_area_unhealthy)
        self.assertTrue(home._auto_area_cancel_event.is_set())
        self.assertTrue(future.cancelled())
        log_signal.emit.assert_called_once()

    def test_stale_processing_status_is_ignored_after_stop(self):
        tasks = mock.Mock()
        home = types.SimpleNamespace(
            _stop_event=threading.Event(),
            _closing=False,
            task_list_component=tasks,
        )
        home._stop_event.set()

        HomeInterface._apply_task_status(
            home,
            4,
            home_interface.TaskStatus.PROCESSING,
        )
        tasks.update_task_status.assert_not_called()

        HomeInterface._apply_task_status(
            home,
            4,
            home_interface.TaskStatus.PENDING,
        )
        tasks.update_task_status.assert_called_once_with(
            4,
            home_interface.TaskStatus.PENDING,
        )

    def test_persistent_worker_is_reused_between_batch_jobs(self):
        home = _LifecycleHarness()
        process = _FakeProcess()
        command_queue = _FakeQueue()
        remote_caller = _FakeRemoteCaller()
        manager = _FakeManager()

        with (
            mock.patch.object(home_interface.multiprocessing, "Process", return_value=process) as ctor,
            mock.patch.object(home_interface.multiprocessing, "Queue", return_value=command_queue),
            mock.patch.object(home_interface, "SubtitleRemoverRemoteCall", return_value=remote_caller),
            mock.patch.object(home_interface.ProcessManager, "instance", return_value=manager),
        ):
            first = home._ensure_subtitle_worker()
            second = home._ensure_subtitle_worker()

        self.assertIs(first, process)
        self.assertIs(second, process)
        self.assertEqual(process.started, 1)
        ctor.assert_called_once()
        self.assertEqual(manager.add_calls, [process])

    def test_closing_gate_prevents_worker_creation(self):
        home = _LifecycleHarness()
        home._closing = True
        with mock.patch.object(home_interface.multiprocessing, "Process") as ctor:
            self.assertIsNone(home._ensure_subtitle_worker())
        ctor.assert_not_called()

    def test_manager_shutdown_race_does_not_publish_started_worker(self):
        home = _LifecycleHarness()
        process = _FakeProcess()
        command_queue = _FakeQueue()
        remote_caller = _FakeRemoteCaller()
        manager = _RejectingManager()

        with (
            mock.patch.object(home_interface.multiprocessing, "Process", return_value=process),
            mock.patch.object(home_interface.multiprocessing, "Queue", return_value=command_queue),
            mock.patch.object(home_interface, "SubtitleRemoverRemoteCall", return_value=remote_caller),
            mock.patch.object(home_interface.ProcessManager, "instance", return_value=manager),
        ):
            with self.assertRaisesRegex(RuntimeError, "shutting down"):
                home._ensure_subtitle_worker()

        self.assertEqual(manager.terminated, [process])
        self.assertIsNone(home.worker_process)
        self.assertIsNone(home.worker_process_id)
        self.assertTrue(remote_caller.stopped)
        self.assertTrue(command_queue.closed)
        self.assertTrue(command_queue.cancelled_join)

    def test_registration_exception_terminates_unpublished_started_worker(self):
        home = _LifecycleHarness()
        process = _FakeProcess()
        command_queue = _FakeQueue()
        remote_caller = _FakeRemoteCaller()
        manager = _FailingRegistrationManager()

        with (
            mock.patch.object(home_interface.multiprocessing, "Process", return_value=process),
            mock.patch.object(home_interface.multiprocessing, "Queue", return_value=command_queue),
            mock.patch.object(home_interface, "SubtitleRemoverRemoteCall", return_value=remote_caller),
            mock.patch.object(home_interface.ProcessManager, "instance", return_value=manager),
        ):
            with self.assertRaisesRegex(RuntimeError, "registration failed"):
                home._ensure_subtitle_worker()

        self.assertEqual(manager.terminated, [process])
        self.assertFalse(process.is_alive())
        self.assertIsNone(home.worker_process)
        self.assertTrue(remote_caller.stopped)
        self.assertTrue(command_queue.closed)

    def test_shutdown_blocks_late_worker_recreation_and_cleans_resources(self):
        home = _LifecycleHarness()
        process = _FakeProcess(alive=True)
        command_queue = _FakeQueue()
        remote_caller = _FakeRemoteCaller()
        manager = _FakeManager()
        home.worker_process = process
        home.worker_process_id = "worker-id"
        home.worker_command_queue = command_queue
        home.worker_remote_caller = remote_caller
        home.running_process = process
        late_result = []

        def late_worker_attempt():
            home._stop_event.wait(timeout=1)
            late_result.append(home._ensure_subtitle_worker())

        worker_thread = threading.Thread(target=late_worker_attempt)
        home._worker_thread = worker_thread
        worker_thread.start()

        with (
            mock.patch.object(home_interface.ProcessManager, "instance", return_value=manager),
            mock.patch.object(home_interface.multiprocessing, "Process") as ctor,
        ):
            home.shutdown_workers(join_timeout=1)
            home.shutdown_workers(join_timeout=1)

        self.assertEqual(late_result, [None])
        ctor.assert_not_called()
        self.assertTrue(home._closing)
        self.assertIsNone(home.worker_process)
        self.assertEqual(manager.terminated, [process])
        self.assertEqual(manager.removed, ["worker-id"])
        self.assertTrue(remote_caller.stopped)
        self.assertTrue(command_queue.closed)
        self.assertTrue(command_queue.cancelled_join)
        home.auto_area_executor.shutdown.assert_called_once_with(
            wait=False, cancel_futures=True
        )
        home.subtitle_timeline_executor.shutdown.assert_called_once_with(
            wait=False, cancel_futures=True
        )

    def test_shutdown_bounded_wait_covers_auto_manual_and_timeline_futures(self):
        home = _LifecycleHarness()
        auto_future = mock.Mock()
        manual_future = mock.Mock()
        timeline_future = mock.Mock()
        home._cancel_auto_area_detections.return_value = [
            auto_future,
            manual_future,
        ]
        home._cancel_subtitle_timeline_prefetch.return_value = [timeline_future]

        with mock.patch.object(home_interface, "wait_futures") as wait:
            home.shutdown_workers(join_timeout=1)

        waited = wait.call_args.args[0]
        self.assertEqual(set(waited), {auto_future, manual_future, timeline_future})
        self.assertGreaterEqual(wait.call_args.kwargs["timeout"], 0)
        self.assertLessEqual(wait.call_args.kwargs["timeout"], 1)

    def test_shutdown_releases_preview_capture(self):
        home = _LifecycleHarness()
        capture = mock.Mock()
        home.video_cap = capture

        home.shutdown_workers(join_timeout=0)
        home.shutdown_workers(join_timeout=0)

        capture.release.assert_called_once_with()
        self.assertIsNone(home.video_cap)

    def test_parent_watchdog_exits_after_parent_join(self):
        parent = mock.Mock()
        exit_func = mock.Mock()
        cleanup_func = mock.Mock()

        self.assertTrue(_exit_after_parent_process(
            parent,
            exit_func=exit_func,
            cleanup_func=cleanup_func,
        ))

        parent.join.assert_called_once_with()
        cleanup_func.assert_called_once_with()
        exit_func.assert_called_once_with(1)

    def test_parent_watchdog_still_exits_when_cleanup_fails(self):
        parent = mock.Mock()
        exit_func = mock.Mock()

        self.assertTrue(_exit_after_parent_process(
            parent,
            exit_func=exit_func,
            cleanup_func=mock.Mock(side_effect=RuntimeError("cleanup failed")),
        ))

        exit_func.assert_called_once_with(1)

    def test_parent_watchdog_thread_detects_parent_exit(self):
        release_parent = threading.Event()
        exited = threading.Event()
        parent = mock.Mock()
        parent.join.side_effect = lambda: release_parent.wait(timeout=1)

        thread = _start_parent_death_watchdog(
            parent_process=parent,
            exit_func=lambda _code: exited.set(),
        )
        self.assertTrue(thread.daemon)
        self.assertFalse(exited.is_set())
        release_parent.set()
        thread.join(timeout=1)
        self.assertTrue(exited.is_set())

    def test_worker_starts_parent_watchdog(self):
        command_queue = mock.Mock()
        command_queue.get.return_value = ("shutdown",)
        manager = _FakeManager()
        with (
            mock.patch.object(home_interface, "_start_parent_death_watchdog") as watchdog,
            mock.patch.object(home_interface.ProcessManager, "instance", return_value=manager),
        ):
            HomeInterface.remover_worker_process(command_queue, mock.Mock())
        watchdog.assert_called_once_with(cleanup_func=manager.shutdown)


if __name__ == "__main__":
    unittest.main()
