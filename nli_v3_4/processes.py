"""Bounded child processes and interruption propagation for local experiments."""
import os
import signal
import subprocess
import threading
import time


class RunInterrupted(KeyboardInterrupt):
    pass


class ChildProcesses:
    def __init__(self):
        self.stopped = threading.Event()
        self.lock = threading.Lock()
        self.children = set()

    def check(self):
        if self.stopped.is_set():
            raise RunInterrupted("experiment interrupted; queued jobs must not start")

    def run(self, cmd, *, cwd, stdout=None, check=False):
        with self.lock:
            self.check()
            process = subprocess.Popen(cmd, cwd=cwd, stdout=stdout,
                stderr=subprocess.STDOUT if stdout is not None else None, start_new_session=True)
            self.children.add(process)
        try:
            code = process.wait()
            self.check()
            if check and code:
                raise subprocess.CalledProcessError(code, cmd)
            return code
        finally:
            with self.lock:
                if process.poll() is not None:
                    self.children.discard(process)

    def stop(self):
        self.stopped.set()
        with self.lock:
            children = list(self.children)
        # Interrupt the actual generation children first, so their finally blocks
        # export their already-flushed journals before the supervisor exits.
        for process in children:
            if process.poll() is None:
                try: os.killpg(process.pid, signal.SIGINT)
                except ProcessLookupError: pass
        deadline = time.monotonic() + 3
        for process in children:
            try: process.wait(timeout=max(.01, deadline-time.monotonic()))
            except subprocess.TimeoutExpired:
                try: os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError: pass
                try: process.wait(timeout=1)
                except subprocess.TimeoutExpired:
                    try: os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError: pass
                    process.wait()

    def signal_handler(self, signum, frame):
        if self.stopped.is_set():
            return
        self.stopped.set()
        raise RunInterrupted(f"experiment interrupted by signal {signum}")
