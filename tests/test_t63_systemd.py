"""Tests for systemd unit file configurations."""

import configparser
import os
from pathlib import Path

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


def test_systemd_units_exist() -> None:
    for name in UNIT_FILES:
        unit_path = DEPLOY_DIR / name
        assert unit_path.is_file(), f"Missing unit file: {unit_path}"


def test_systemd_unit_structure_and_executables() -> None:
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
        assert parser.get("Service", "WorkingDirectory") == "/home/judas/bmo-assistant"

        exec_start = parser.get("Service", "ExecStart")
        binary = exec_start.split()[0]
        assert binary == "/home/judas/bmo-assistant/.venv/bin/python"
        assert os.path.isfile(binary), f"Binary does not exist: {binary}"
        assert os.access(binary, os.X_OK), f"Binary is not executable: {binary}"


def test_service_dependencies() -> None:
    for name in DEPENDENT_SERVICES:
        unit_path = DEPLOY_DIR / name
        parser = configparser.ConfigParser(interpolation=None)
        with open(unit_path, encoding="utf-8") as f:
            parser.read_file(f)

        after_val = parser.get("Unit", "After")
        assert "bmo-broker.service" in after_val.split(), f"{name} does not specify bmo-broker.service in After"
