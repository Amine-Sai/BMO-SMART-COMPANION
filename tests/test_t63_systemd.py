"""Tests for systemd unit file configurations."""

import configparser
import os
from pathlib import Path

from tools.simple_broker import parse_args

REPO_ROOT = Path(__file__).resolve().parent.parent
DEPLOY_DIR = REPO_ROOT / "deploy" / "systemd"

UNIT_FILES = [
    "bmo-broker.service",
    "bmo-cv.service",
    "bmo-brain.service",
    "bmo-pi.service",
]

DEPENDENT_SERVICES = [
    "bmo-cv.service",
    "bmo-brain.service",
    "bmo-pi.service",
]

HARDWARE_SERVICES = [
    "bmo-cv.service",
    "bmo-brain.service",
    "bmo-pi.service",
]


def test_systemd_units_exist() -> None:
    for name in UNIT_FILES:
        unit_path = DEPLOY_DIR / name
        assert unit_path.is_file(), f"Missing unit file: {unit_path}"


def test_systemd_unit_structure_and_executables() -> None:
    home_str = str(Path.home())
    repo_str = str(REPO_ROOT)

    for name in UNIT_FILES:
        unit_path = DEPLOY_DIR / name
        parser = configparser.ConfigParser(interpolation=None)
        with open(unit_path, encoding="utf-8") as f:
            parser.read_file(f)

        assert parser.has_section("Unit")
        assert parser.has_section("Service")
        assert parser.has_section("Install")

        assert parser.get("Service", "Restart") == "on-failure"
        assert parser.get("Service", "RestartSec") == "3"
        assert parser.get("Service", "NoNewPrivileges") == "yes"

        work_dir = parser.get("Service", "WorkingDirectory")
        assert "%h" in work_dir
        expanded_work_dir = work_dir.replace("%h", home_str)
        assert expanded_work_dir == repo_str

        exec_start = parser.get("Service", "ExecStart")
        raw_binary = exec_start.split()[0]
        expanded_binary = raw_binary.replace("%h", home_str)

        if expanded_binary.startswith(repo_str) or expanded_binary.startswith(home_str):
            # Only isfile-check repo-local executables
            assert os.path.isfile(expanded_binary), f"Binary does not exist: {expanded_binary}"
            assert os.access(expanded_binary, os.X_OK), f"Binary is not executable: {expanded_binary}"
        else:
            # System binary (e.g. /usr/sbin/mosquitto) - must not require binary to be installed on test host
            assert expanded_binary == "/usr/sbin/mosquitto"


def test_bmo_broker_exec_start() -> None:
    unit_path = DEPLOY_DIR / "bmo-broker.service"
    parser = configparser.ConfigParser(interpolation=None)
    with open(unit_path, encoding="utf-8") as f:
        parser.read_file(f)

    exec_start = parser.get("Service", "ExecStart")
    parts = exec_start.split()
    assert parts[0] == "/usr/sbin/mosquitto"
    assert "-c" in parts
    conf_idx = parts.index("-c") + 1
    assert parts[conf_idx] == "%h/bmo-assistant/infra/mosquitto.conf"


def test_hardware_services_supplementary_groups() -> None:
    for name in HARDWARE_SERVICES:
        unit_path = DEPLOY_DIR / name
        parser = configparser.ConfigParser(interpolation=None)
        with open(unit_path, encoding="utf-8") as f:
            parser.read_file(f)

        groups = parser.get("Service", "SupplementaryGroups").split()
        assert "audio" in groups
        assert "video" in groups


def test_bmo_pi_environment_file() -> None:
    unit_path = DEPLOY_DIR / "bmo-pi.service"
    parser = configparser.ConfigParser(interpolation=None)
    with open(unit_path, encoding="utf-8") as f:
        parser.read_file(f)

    env_file = parser.get("Service", "EnvironmentFile")
    assert env_file == "-%h/.config/bmo/pi.env"


def test_service_dependencies() -> None:
    for name in DEPENDENT_SERVICES:
        unit_path = DEPLOY_DIR / name
        parser = configparser.ConfigParser(interpolation=None)
        with open(unit_path, encoding="utf-8") as f:
            parser.read_file(f)

        after_val = parser.get("Unit", "After")
        assert "bmo-broker.service" in after_val.split(), f"{name} does not specify bmo-broker.service in After"


def test_mosquitto_conf_requirements() -> None:
    conf_path = REPO_ROOT / "infra" / "mosquitto.conf"
    assert conf_path.is_file(), f"Missing conf file: {conf_path}"
    content = conf_path.read_text("utf-8")
    assert "allow_anonymous false" in content
    assert "password_file" in content
    assert "acl_file" in content
    assert "mosquitto_passwd" in content


def test_simple_broker_honours_host_and_port() -> None:
    args = parse_args(["--host", "127.0.0.1", "--port", "1883"])
    assert args.host == "127.0.0.1"
    assert args.port == 1883

    custom_args = parse_args(["--host", "0.0.0.0", "--port", "1884"])
    assert custom_args.host == "0.0.0.0"
    assert custom_args.port == 1884
