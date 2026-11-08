from pathlib import Path

from edge.esp32_protocol import (
    ESP32ProtocolSimulator,
    format_move_cmd,
    parse_esp32_response,
)
from tools.gen_limits_header import generate_limits_header


def test_limits_header_consistency() -> None:
    expected = generate_limits_header()
    actual = Path("edge/esp32_firmware/limits.h").read_text(encoding="utf-8")
    assert actual == expected


def test_firmware_file_contents() -> None:
    content = Path("edge/esp32_firmware/bmo_servo_controller.ino").read_text(encoding="utf-8")
    assert '#include "limits.h"' in content
    assert "SERVO_MAX_DEG_PER_SEC" in content
    assert "SERVO_WATCHDOG_TIMEOUT_MS" in content


def test_protocol_format_and_parse() -> None:
    cmd = format_move_cmd("pan", 45.0)
    assert cmd == "MOVE pan 45.0\n"

    parsed_ok = parse_esp32_response("OK pan 60.0\n")
    assert parsed_ok["ok"] is True
    assert parsed_ok["servo_id"] == "pan"
    assert parsed_ok["angle"] == 60.0

    parsed_pong = parse_esp32_response("PONG\n")
    assert parsed_pong["ok"] is True
    assert parsed_pong["status"] == "pong"

    parsed_stop = parse_esp32_response("OK STOP\n")
    assert parsed_stop["ok"] is True
    assert parsed_stop["command"] == "STOP"

    parsed_err = parse_esp32_response("ERR unknown_servo\n")
    assert parsed_err["ok"] is False
    assert parsed_err["error"] == "unknown_servo"

    parsed_empty = parse_esp32_response("")
    assert parsed_empty["ok"] is False


def test_simulator_boot_and_move_clamping() -> None:
    sim = ESP32ProtocolSimulator()
    for name in ("pan", "tilt", "left_arm", "right_arm"):
        assert sim.servos[name].current_deg == 0.0
        assert sim.servos[name].target_deg == 0.0
        assert sim.servos[name].attached is True

    res = sim.handle_line("MOVE pan 500")
    assert res == "OK pan 60.0\n"
    assert sim.servos["pan"].target_deg == 60.0

    err_res = sim.handle_line("MOVE unknown 10")
    assert err_res.startswith("ERR")

    invalid_res = sim.handle_line("MOVE pan")
    assert invalid_res.startswith("ERR")

    invalid_num = sim.handle_line("MOVE pan abc")
    assert invalid_num.startswith("ERR")

    empty_res = sim.handle_line("")
    assert empty_res.startswith("ERR")

    unknown_cmd = sim.handle_line("FOO")
    assert unknown_cmd.startswith("ERR")


def test_simulator_slew_rate() -> None:
    sim = ESP32ProtocolSimulator()
    sim.handle_line("MOVE pan 60.0")
    assert sim.servos["pan"].current_deg == 0.0

    sim.step(0.5)
    assert sim.servos["pan"].current_deg == 45.0

    sim.step(0.5)
    assert sim.servos["pan"].current_deg == 60.0


def test_simulator_watchdog_and_reattach() -> None:
    now = 100.0
    sim = ESP32ProtocolSimulator(clock=lambda: now)
    sim.handle_line("MOVE pan 30.0", now_s=now)

    now += 2.0
    sim.step(0.1, now_s=now)
    assert sim.servos["pan"].attached is True

    now += 1.5
    sim.step(0.1, now_s=now)
    for s in sim.servos.values():
        assert s.attached is False

    sim.handle_line("MOVE pan 10.0", now_s=now)
    assert sim.servos["pan"].attached is True

    now += 4.0
    sim.step(0.1, now_s=now)
    assert sim.servos["pan"].attached is False

    sim.handle_line("PING", now_s=now)
    assert sim.servos["pan"].attached is True

    sim.handle_line("STOP", now_s=now)
    assert sim.servos["pan"].attached is False
