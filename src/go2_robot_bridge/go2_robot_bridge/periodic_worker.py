"""Periodic executor tasks that cannot be starved by ROS subscriptions."""

import threading
import time
from typing import Any, Callable, Optional


class PeriodicScheduler:
    """Queue at most one deadline task from a dedicated scheduler thread."""

    def __init__(
        self,
        period_sec: float,
        schedule: Callable[[Callable[[], None]], Any],
        work: Callable[[], None],
        keep_running: Callable[[], bool],
        on_failure: Callable[[], None],
        name: str,
    ) -> None:
        self._period_sec = float(period_sec)
        if self._period_sec <= 0.0:
            raise ValueError("period_sec must be positive")
        self._schedule = schedule
        self._work = work
        self._keep_running = keep_running
        self._on_failure = on_failure
        self._stop = threading.Event()
        self._pending = threading.Event()
        self._failure: Optional[BaseException] = None
        self._thread = threading.Thread(target=self._run, name=name, daemon=False)

    def start(self) -> None:
        self._thread.start()

    def _run(self) -> None:
        next_run = time.monotonic()
        try:
            while not self._stop.is_set() and self._keep_running():
                remaining = next_run - time.monotonic()
                if remaining > 0.0:
                    self._stop.wait(remaining)
                    continue
                if not self._pending.is_set():
                    self._pending.set()
                    try:
                        self._schedule(self._run_once)
                    except BaseException:
                        self._pending.clear()
                        raise
                next_run = time.monotonic() + self._period_sec
        except BaseException as error:
            self._failure = error
            try:
                self._on_failure()
            except BaseException:
                pass

    def _run_once(self) -> None:
        try:
            self._work()
        except BaseException as error:
            self._failure = error
            try:
                self._on_failure()
            except BaseException:
                pass
        finally:
            self._pending.clear()

    def stop(self, timeout_sec: float = 5.0) -> None:
        self._stop.set()
        self._thread.join(timeout=max(0.0, float(timeout_sec)))
        if self._thread.is_alive():
            raise RuntimeError("periodic scheduler did not stop")

    def raise_if_failed(self) -> None:
        if self._failure is not None:
            raise RuntimeError("periodic executor task failed") from self._failure
