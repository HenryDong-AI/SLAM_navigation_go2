import threading
import time
import unittest

from go2_robot_bridge.periodic_worker import PeriodicScheduler


class PeriodicSchedulerTests(unittest.TestCase):
    def test_periodic_work_is_coalesced_until_executor_runs_it(self):
        done = threading.Event()
        calls = []
        scheduled = []

        def work():
            calls.append(len(calls))
            if len(calls) == 3:
                done.set()

        worker = PeriodicScheduler(
            period_sec=0.01,
            schedule=scheduled.append,
            work=work,
            keep_running=lambda: not done.is_set(),
            on_failure=lambda: None,
            name="periodic-worker-test",
        )
        worker.start()
        limit = time.monotonic() + 0.5
        while len(scheduled) < 1 and time.monotonic() < limit:
            time.sleep(0.001)
        self.assertEqual(len(scheduled), 1)
        time.sleep(0.03)
        self.assertEqual(len(scheduled), 1)
        while not done.is_set() and time.monotonic() < limit:
            scheduled.pop(0)()
            time.sleep(0.02)
        self.assertTrue(done.is_set())
        worker.stop()
        worker.raise_if_failed()
        self.assertEqual(len(calls), 3)

    def test_failure_is_reported(self):
        failed = threading.Event()

        def work():
            raise ValueError("boom")

        worker = PeriodicScheduler(
            period_sec=0.01,
            schedule=lambda callback: callback(),
            work=work,
            keep_running=lambda: True,
            on_failure=failed.set,
            name="periodic-worker-failure-test",
        )
        worker.start()
        self.assertTrue(failed.wait(0.5))
        worker.stop()
        with self.assertRaisesRegex(RuntimeError, "periodic executor task failed"):
            worker.raise_if_failed()


if __name__ == "__main__":
    unittest.main()
