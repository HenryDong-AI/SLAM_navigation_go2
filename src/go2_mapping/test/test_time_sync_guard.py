import unittest

from go2_mapping.time_sync_guard import TimeSyncStatusGuard


class TimeSyncStatusGuardTest(unittest.TestCase):
    @staticmethod
    def status(state="locked", instance_id="bridge-a", epoch=0, **extra):
        result = {
            "state": state,
            "instance_id": instance_id,
            "epoch": epoch,
        }
        result.update(extra)
        return result

    def test_optional_guard_is_always_ready(self):
        guard = TimeSyncStatusGuard(required=False)
        self.assertTrue(guard.ready)
        self.assertEqual(guard.update({}), "")

    def test_warmup_to_locked(self):
        guard = TimeSyncStatusGuard()
        self.assertEqual(guard.update(self.status(state="warming")), "")
        self.assertFalse(guard.ready)
        self.assertEqual(guard.update(self.status()), "")
        self.assertTrue(guard.ready)

    def test_fault_epoch_instance_and_unlock_latch(self):
        changes = (
            self.status(state="fault_latched", fault_reason="reset"),
            self.status(epoch=1),
            self.status(instance_id="bridge-b"),
            self.status(state="warming"),
        )
        for changed in changes:
            guard = TimeSyncStatusGuard()
            guard.update(self.status())
            self.assertTrue(guard.update(changed))
            self.assertFalse(guard.ready)


if __name__ == "__main__":
    unittest.main()
