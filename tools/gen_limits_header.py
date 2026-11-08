"""Generate C/C++ header for ESP32 firmware from config/limits.yaml."""

from pathlib import Path
import sys
from typing import Any
import yaml


def generate_limits_header(limits_path: str | Path = "config/limits.yaml") -> str:
    path = Path(limits_path)
    with path.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f)

    servos = data.get("servos", {})
    rate = data.get("rate_limits", {})

    pan = servos.get("pan", {})
    tilt = servos.get("tilt", {})
    left = servos.get("left_arm", {})
    right = servos.get("right_arm", {})

    lines = [
        "/* Auto-generated from config/limits.yaml - DO NOT EDIT MANUALLY */",
        "#ifndef BMO_LIMITS_H",
        "#define BMO_LIMITS_H",
        "",
        f"#define SERVO_PAN_MIN_DEG {float(pan.get('min_deg', -60.0)):.1f}f",
        f"#define SERVO_PAN_MAX_DEG {float(pan.get('max_deg', 60.0)):.1f}f",
        f"#define SERVO_PAN_BOOT_DEG {float(pan.get('boot_deg', 0.0)):.1f}f",
        "",
        f"#define SERVO_TILT_MIN_DEG {float(tilt.get('min_deg', -30.0)):.1f}f",
        f"#define SERVO_TILT_MAX_DEG {float(tilt.get('max_deg', 45.0)):.1f}f",
        f"#define SERVO_TILT_BOOT_DEG {float(tilt.get('boot_deg', 0.0)):.1f}f",
        "",
        f"#define SERVO_LEFT_ARM_MIN_DEG {float(left.get('min_deg', 0.0)):.1f}f",
        f"#define SERVO_LEFT_ARM_MAX_DEG {float(left.get('max_deg', 120.0)):.1f}f",
        f"#define SERVO_LEFT_ARM_BOOT_DEG {float(left.get('boot_deg', 0.0)):.1f}f",
        "",
        f"#define SERVO_RIGHT_ARM_MIN_DEG {float(right.get('min_deg', 0.0)):.1f}f",
        f"#define SERVO_RIGHT_ARM_MAX_DEG {float(right.get('max_deg', 120.0)):.1f}f",
        f"#define SERVO_RIGHT_ARM_BOOT_DEG {float(right.get('boot_deg', 0.0)):.1f}f",
        "",
        f"#define SERVO_MAX_DEG_PER_SEC {float(rate.get('max_deg_per_sec', 90.0)):.1f}f",
        f"#define SERVO_WATCHDOG_TIMEOUT_MS {int(rate.get('watchdog_timeout_ms', 3000))}",
        "",
        "#endif /* BMO_LIMITS_H */\n",
    ]
    return "\n".join(lines)


def main() -> None:
    header = generate_limits_header()
    if "--stdout" in sys.argv:
        sys.stdout.write(header)
        return
    out_path = Path("edge/esp32_firmware/limits.h")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(header, encoding="utf-8")


if __name__ == "__main__":
    main()
