from collections.abc import Callable
from dataclasses import dataclass
import time
from typing import Any

from brain.tools.actuator_tools import load_limits


def format_move_cmd(servo_id: str, angle: float) -> str:
    return f"MOVE {servo_id} {angle:.1f}\n"


def parse_esp32_response(line: str) -> dict[str, Any]:
    cleaned = line.strip()
    if not cleaned:
        return {"ok": False, "status": "empty"}
    parts = cleaned.split()
    cmd = parts[0]
    if cmd == "PONG":
        return {"ok": True, "status": "pong"}
    if cmd == "OK":
        if len(parts) == 2 and parts[1] == "STOP":
            return {"ok": True, "status": "ok", "command": "STOP"}
        if len(parts) >= 3:
            try:
                return {
                    "ok": True,
                    "status": "ok",
                    "servo_id": parts[1],
                    "angle": float(parts[2]),
                }
            except ValueError:
                pass
        return {"ok": True, "status": "ok", "args": parts[1:]}
    if cmd == "ERR":
        reason = " ".join(parts[1:]) if len(parts) > 1 else "unknown"
        return {"ok": False, "status": "err", "error": reason}
    return {"ok": False, "status": "unknown", "raw": cleaned}


def _extract_servo_bounds(limits: Any, servo_id: str) -> tuple[float, float]:
    servo_conf = None
    if hasattr(limits, "servos"):
        servos_field = limits.servos
        if isinstance(servos_field, dict):
            servo_conf = servos_field.get(servo_id)
        else:
            servo_conf = getattr(servos_field, servo_id, None)
    if servo_conf is None and hasattr(limits, servo_id):
        servo_conf = getattr(limits, servo_id)

    if servo_conf is not None:
        if isinstance(servo_conf, dict):
            return float(servo_conf["min_deg"]), float(servo_conf["max_deg"])
        return float(servo_conf.min_deg), float(servo_conf.max_deg)

    defaults = {
        "pan": (-60.0, 60.0),
        "tilt": (-30.0, 45.0),
        "left_arm": (0.0, 120.0),
        "right_arm": (0.0, 120.0),
    }
    return defaults.get(servo_id, (-90.0, 90.0))


@dataclass
class ServoState:
    min_deg: float
    max_deg: float
    current_deg: float = 0.0
    target_deg: float = 0.0
    attached: bool = True


class ESP32ProtocolSimulator:
    def __init__(
        self,
        limits_path: str = "config/limits.yaml",
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._clock = clock
        self.limits = load_limits(limits_path)
        self.max_deg_per_sec: float = float(self.limits.max_deg_per_sec)
        self.watchdog_timeout_ms: float = float(self.limits.watchdog_timeout_ms)

        servo_ids = ("pan", "tilt", "left_arm", "right_arm")
        self.servos: dict[str, ServoState] = {}
        for s_id in servo_ids:
            min_deg, max_deg = _extract_servo_bounds(self.limits, s_id)
            self.servos[s_id] = ServoState(
                min_deg=min_deg,
                max_deg=max_deg,
                current_deg=0.0,
                target_deg=0.0,
                attached=True,
            )
        self._last_cmd_s: float = self._clock()

    def handle_line(self, line: str, now_s: float | None = None) -> str:
        now = self._clock() if now_s is None else now_s
        cmd = line.strip()
        if not cmd:
            return "ERR empty_command\n"

        if cmd == "PING":
            self._last_cmd_s = now
            for s in self.servos.values():
                s.attached = True
            return "PONG\n"

        if cmd == "STOP":
            for s in self.servos.values():
                s.attached = False
            return "OK STOP\n"

        if cmd.startswith("MOVE"):
            parts = cmd.split()
            if len(parts) != 3:
                return "ERR invalid_args\n"
            servo_id = parts[1]
            try:
                angle = float(parts[2])
            except ValueError:
                return "ERR invalid_args\n"

            if servo_id not in self.servos:
                return "ERR unknown_servo\n"

            servo = self.servos[servo_id]
            clamped = max(servo.min_deg, min(servo.max_deg, angle))
            servo.target_deg = clamped
            servo.attached = True
            self._last_cmd_s = now
            return f"OK {servo_id} {clamped:.1f}\n"

        return "ERR unknown_command\n"

    def step(self, dt_s: float, now_s: float | None = None) -> None:
        now = self._clock() if now_s is None else now_s
        if (now - self._last_cmd_s) * 1000.0 > self.watchdog_timeout_ms:
            for s in self.servos.values():
                s.attached = False

        max_step = self.max_deg_per_sec * dt_s
        for s in self.servos.values():
            if s.attached:
                diff = s.target_deg - s.current_deg
                if abs(diff) <= max_step:
                    s.current_deg = s.target_deg
                elif diff > 0:
                    s.current_deg += max_step
                else:
                    s.current_deg -= max_step
