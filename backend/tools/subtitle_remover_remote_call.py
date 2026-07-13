import multiprocessing
import queue as queue_module
import threading
import traceback
from enum import Enum


class Command(Enum):
    JOB_FINISH = 0
    PROGRESS = 1
    LOG = 2
    MANAGE_PROCESS = 3
    ERROR = 4
    UPDATE_PREVIEW_WITH_COMP = 5
    SHUTDOWN = 6
    PROCESSING_PHASE = 7


class SubtitleRemoverRemoteCall:
    """跨进程回调分发器。"""

    def __init__(self, queue=None):
        self.queue = queue or multiprocessing.Queue()
        self.callbacks = {}
        self.running = True
        self.job_finished_event = threading.Event()
        self.job_had_error = False
        self._thread = threading.Thread(target=self.run, daemon=True)
        self._thread.start()

    def run(self):
        try:
            while self.running:
                try:
                    cmd, args = self.queue.get(timeout=0.2)
                except queue_module.Empty:
                    continue
                if cmd == Command.SHUTDOWN:
                    break
                if cmd == Command.JOB_FINISH:
                    self.job_finished_event.set()
                    continue
                if cmd == Command.ERROR:
                    self.job_had_error = True
                callback = self.callbacks.get(cmd)
                if callback:
                    try:
                        callback(*args)
                    except Exception:
                        # A UI callback must not kill the dispatcher and leave
                        # the worker/job-finished handshake permanently stuck.
                        traceback.print_exc()
        finally:
            self.running = False
            self.job_finished_event.set()

    def stop(self):
        self.running = False
        try:
            self.queue.put_nowait((Command.SHUTDOWN, ()))
        except Exception:
            pass
        if self._thread.is_alive():
            self._thread.join(timeout=1)

    def reset_job_state(self):
        self.job_had_error = False
        self.job_finished_event.clear()

    def wait_for_job_finish(self, timeout=None):
        return self.job_finished_event.wait(timeout=timeout)

    def register_update_progress_callback(self, callback):
        self.callbacks[Command.PROGRESS] = callback

    def register_log_callback(self, callback):
        self.callbacks[Command.LOG] = callback

    def register_manage_process_callback(self, callback):
        self.callbacks[Command.MANAGE_PROCESS] = callback

    def register_update_preview_with_comp_callback(self, callback):
        self.callbacks[Command.UPDATE_PREVIEW_WITH_COMP] = callback

    def register_error_callback(self, callback):
        self.callbacks[Command.ERROR] = callback

    def register_processing_phase_callback(self, callback):
        self.callbacks[Command.PROCESSING_PHASE] = callback

    @staticmethod
    def remote_call_update_progress(queue, progress, isFinished):
        queue.put((Command.PROGRESS, (progress, isFinished)))

    @staticmethod
    def remote_call_append_log(queue, *args):
        queue.put((Command.LOG, (*args,)))

    @staticmethod
    def remote_call_finish_job(queue):
        queue.put((Command.JOB_FINISH, ()))

    @staticmethod
    def remote_call_finish(queue, *args):
        SubtitleRemoverRemoteCall.remote_call_finish_job(queue)

    @staticmethod
    def remote_call_catch_error(queue, e):
        queue.put((Command.ERROR, (e,)))

    @staticmethod
    def remote_call_manage_process(queue, pid):
        queue.put((Command.MANAGE_PROCESS, (pid,)))

    @staticmethod
    def remote_call_update_preview_with_comp(queue, *args):
        queue.put((Command.UPDATE_PREVIEW_WITH_COMP, (*args,)))

    @staticmethod
    def remote_call_processing_phase(queue, phase, video_path):
        queue.put((Command.PROCESSING_PHASE, (phase, video_path)))
