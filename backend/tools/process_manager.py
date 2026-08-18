# -*- coding: utf-8 -*-
"""Thread-safe child-process lifecycle management.

The GUI keeps a persistent ``multiprocessing.Process`` alive between batch
items.  During application shutdown no new process may be registered, and a
worker must be terminated together with descendants such as FFmpeg.  Keeping
those two rules here avoids subtly different cleanup sequences in UI code.
"""

import atexit
import concurrent.futures
import logging
import os
import platform
import signal
import subprocess
import threading
import time

from .subprocess_utils import hidden_subprocess_kwargs


class ProcessManager:
    """Manage subprocesses and multiprocessing workers owned by the GUI."""

    _instance = None
    _instance_lock = threading.Lock()

    @classmethod
    def instance(cls):
        with cls._instance_lock:
            if cls._instance is None:
                cls._instance = cls()
            return cls._instance

    def __init__(self):
        self.processes = {}
        self.logger = logging.getLogger(__name__)
        self._lock = threading.RLock()
        self._accepting_processes = True
        atexit.register(self.shutdown)

    @property
    def accepting_processes(self):
        with self._lock:
            return self._accepting_processes

    def begin_shutdown(self):
        """Permanently reject process registration for this app instance."""
        with self._lock:
            self._accepting_processes = False

    def add_process(self, process, name=None):
        """Register a process, or immediately clean it up during shutdown.

        Returning ``None`` tells the caller that shutdown won the race and the
        process must not be published as a usable worker.
        """
        if process is None:
            return None
        process_id = name or f"Process:{id(process)}"
        with self._lock:
            if self._accepting_processes:
                self.processes[process_id] = process
                return process_id
        self.terminate_by_process(process)
        return None

    def add_pid(self, pid, name=None):
        if not self._valid_pid(pid):
            return None
        process_id = name or f"Pid:{pid}"
        with self._lock:
            if self._accepting_processes:
                self.processes[process_id] = int(pid)
                return process_id
        self.terminate_by_pid(int(pid))
        return None

    def remove_process(self, process_id):
        with self._lock:
            return self.processes.pop(process_id, None) is not None

    def _terminate_managed_process(self, process):
        """Terminate one registered entry without aborting cleanup of others."""
        try:
            if isinstance(process, int):
                return self.terminate_by_pid(process)
            return self.terminate_by_process(process)
        except Exception as exc:
            self.logger.warning("Failed to terminate managed process: %s", exc)
            return False

    def _terminate_synchronously(self, processes):
        """Terminate entries without creating threads (safe during atexit)."""
        for process in processes:
            self._terminate_managed_process(process)

    def terminate_all(self, parallel=True):
        """Atomically detach and terminate all registered process trees.

        Explicit runtime cleanup remains concurrent by default.  Interpreter
        shutdown cannot create or submit new ``ThreadPoolExecutor`` work, so
        callers may request synchronous cleanup and executor failures also
        fall back to terminating every not-yet-submitted entry in sequence.
        """
        with self._lock:
            processes = list(self.processes.values())
            self.processes.clear()
        if not processes:
            return

        if not parallel:
            self._terminate_synchronously(processes)
            return

        executor = None
        futures = []
        submitted_count = 0
        try:
            executor = concurrent.futures.ThreadPoolExecutor(
                max_workers=min(8, len(processes))
            )
            for process in processes:
                futures.append(
                    executor.submit(self._terminate_managed_process, process)
                )
                submitted_count += 1
            concurrent.futures.wait(futures)
        except Exception as exc:
            self.logger.warning(
                "Concurrent process cleanup unavailable; falling back to "
                "synchronous termination: %s",
                exc,
            )
            if futures:
                try:
                    concurrent.futures.wait(futures)
                except Exception:
                    pass
            self._terminate_synchronously(processes[submitted_count:])
        finally:
            if executor is not None:
                try:
                    executor.shutdown(wait=True)
                except Exception:
                    pass

    def shutdown(self):
        """Close registration first so cleanup cannot race a late process add."""
        self.begin_shutdown()
        # ``shutdown`` is also registered with atexit.  At that point Python's
        # concurrent.futures module has already disabled new executor work, so
        # this path must never depend on creating helper threads.
        self.terminate_all(parallel=False)

    @staticmethod
    def _valid_pid(pid):
        try:
            pid = int(pid)
        except (TypeError, ValueError):
            return False
        return pid > 1 and pid != os.getpid()

    @staticmethod
    def _process_is_alive(process):
        try:
            if hasattr(process, "is_alive"):
                return bool(process.is_alive())
            if hasattr(process, "poll"):
                return process.poll() is None
        except (AssertionError, OSError, ValueError):
            return False
        return True

    @staticmethod
    def _wait_for_process(process, timeout):
        try:
            if hasattr(process, "join"):
                process.join(timeout=timeout)
            elif hasattr(process, "wait"):
                process.wait(timeout=timeout)
        except Exception:
            pass

    def terminate_by_process(self, process):
        """Terminate descendants before their parent can orphan them."""
        if process is None or not self._process_is_alive(process):
            return True
        pid = getattr(process, "pid", None)
        tree_terminated = False
        if self._valid_pid(pid):
            tree_terminated = self.terminate_by_pid(pid)
            self._wait_for_process(process, timeout=1.0)
        if not self._process_is_alive(process):
            return True
        if not tree_terminated:
            self.logger.warning(
                "Process-tree termination failed for %s; falling back to parent termination",
                pid,
            )
        try:
            process.terminate()
        except Exception:
            pass
        self._wait_for_process(process, timeout=1.0)
        if not self._process_is_alive(process):
            return True
        try:
            process.kill()
        except Exception:
            pass
        self._wait_for_process(process, timeout=1.0)
        return not self._process_is_alive(process)

    @staticmethod
    def _posix_process_tree(root_pid):
        """Return descendants deepest-first, followed by ``root_pid``."""
        try:
            result = subprocess.run(
                ["ps", "-Ao", "pid=,ppid="],
                stdout=subprocess.PIPE,
                stderr=subprocess.DEVNULL,
                text=True,
                timeout=2,
            )
            children = {}
            for line in result.stdout.splitlines():
                fields = line.split()
                if len(fields) != 2:
                    continue
                pid, parent_pid = map(int, fields)
                children.setdefault(parent_pid, []).append(pid)
        except Exception:
            return [root_pid]

        ordered = []
        visited = set()

        def visit(pid):
            if pid in visited:
                return
            visited.add(pid)
            for child_pid in children.get(pid, ()):
                visit(child_pid)
            ordered.append(pid)

        visit(root_pid)
        return ordered

    @staticmethod
    def _signal_pids(pids, sig):
        for pid in pids:
            try:
                os.kill(pid, sig)
            except (ProcessLookupError, PermissionError, OSError):
                pass

    def terminate_by_pid(self, pid):
        """Forcefully terminate ``pid`` and all descendants cross-platform."""
        if not self._valid_pid(pid):
            return False
        pid = int(pid)
        try:
            if platform.system() == "Windows":
                for attempt in range(2):
                    try:
                        result = subprocess.run(
                            ["taskkill", "/F", "/T", "/PID", str(pid)],
                            stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE,
                            timeout=5,
                            **hidden_subprocess_kwargs(),
                        )
                    except subprocess.TimeoutExpired:
                        self.logger.warning(
                            "taskkill timed out for process tree %s (attempt %s/2)",
                            pid,
                            attempt + 1,
                        )
                        continue
                    if result.returncode == 0:
                        return True
                    self.logger.warning(
                        "taskkill failed for process tree %s with exit code %s (attempt %s/2)",
                        pid,
                        result.returncode,
                        attempt + 1,
                    )
                return False

            process_tree = self._posix_process_tree(pid)
            self._signal_pids(process_tree, signal.SIGTERM)
            deadline = time.monotonic() + 1.0
            while time.monotonic() < deadline:
                if not any(self._pid_exists(item) for item in process_tree):
                    return True
                time.sleep(0.05)
            self._signal_pids(process_tree, signal.SIGKILL)
            deadline = time.monotonic() + 0.5
            while time.monotonic() < deadline:
                if not any(self._pid_exists(item) for item in process_tree):
                    return True
                time.sleep(0.05)
            return not any(self._pid_exists(item) for item in process_tree)
        except Exception as exc:
            self.logger.warning("Failed to terminate process tree %s: %s", pid, exc)
            return False

    @staticmethod
    def _pid_exists(pid):
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return False
        except PermissionError:
            return True
        except OSError:
            return False
        return True
