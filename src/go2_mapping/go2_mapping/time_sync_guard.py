"""Pure downstream guard for the sensor time boundary status stream."""

from typing import Mapping, Optional


class TimeSyncStatusGuard:
    """Accept one bridge instance/epoch and latch on any discontinuity."""

    def __init__(self, required: bool = True) -> None:
        self.required = bool(required)
        self.instance_id: Optional[str] = None
        self.epoch: Optional[int] = None
        self.state = "unseen"
        self.ever_locked = False
        self.fault_reason = ""

    @property
    def ready(self) -> bool:
        return not self.required or (
            self.state == "locked" and not self.fault_reason
        )

    def latch(self, reason: str) -> str:
        if not self.fault_reason:
            self.fault_reason = str(reason)
        self.state = "fault_latched"
        return self.fault_reason

    def update(self, payload: Mapping[str, object]) -> str:
        if not self.required:
            return ""
        if self.fault_reason:
            return self.fault_reason
        if not isinstance(payload, Mapping):
            return self.latch("time-sync status is not an object")
        state = str(payload.get("state", ""))
        instance_id = str(payload.get("instance_id", "")).strip()
        try:
            epoch = int(payload.get("epoch"))
        except (TypeError, ValueError):
            return self.latch("time-sync status has an invalid epoch")
        if state not in ("warming", "locked", "fault_latched"):
            return self.latch("time-sync status has an invalid state")
        if not instance_id or epoch < 0:
            return self.latch("time-sync status identity is invalid")
        if self.instance_id is None:
            self.instance_id = instance_id
            self.epoch = epoch
        elif instance_id != self.instance_id:
            return self.latch("sensor time bridge process changed")
        if state == "fault_latched":
            return self.latch(
                str(payload.get("fault_reason", "")).strip()
                or "sensor time bridge reported a fault"
            )
        if epoch != self.epoch:
            return self.latch("sensor time epoch changed")
        if self.ever_locked and state != "locked":
            return self.latch("sensor time bridge left the locked state")
        self.state = state
        if state == "locked":
            self.ever_locked = True
        return ""
