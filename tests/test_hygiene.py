"""Hygiene tests for BMO repository integrity."""

import pathlib
import re
import subprocess
import tokenize
import unittest
import yaml

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


class TestRepoHygiene(unittest.TestCase):
    """Hygiene verification suite ensuring strict zero-drift repository constraints."""

    def test_zero_bmo_literals_outside_topics(self):
        """Verify zero topic URI string literals outside topics registry."""
        ignored_dirs = {".venv", ".git", "__pycache__", "build", "egg-info", ".eggs"}
        violating_files: list[str] = []
        target_prefix = "bmo" + "/"

        for py_path in REPO_ROOT.rglob("*.py"):
            parts = py_path.parts
            if any(part in ignored_dirs or part.endswith(".egg-info") for part in parts):
                continue
            if py_path.name == "topics.py" and "bmo_contracts" in parts:
                continue

            with open(py_path, "rb") as f:
                try:
                    for tok in tokenize.tokenize(f.readline):
                        if tok.type == tokenize.STRING:
                            # Strip string quotes and check literal content
                            try:
                                val = eval(tok.string)
                                if isinstance(val, str) and (
                                    val.startswith(target_prefix) or (" " + target_prefix) in val
                                ):
                                    violating_files.append(f"{py_path.relative_to(REPO_ROOT)}:{tok.start[0]}")
                            except Exception:
                                pass
                except tokenize.TokenError:
                    continue

        self.assertEqual(violating_files, [], f"Found topic string literals outside topics.py: {violating_files}")

    def test_zero_venv_brain_and_python312_in_tracked_code(self):
        """Zero occurrences of .venv-brain or python3.12 in tracked code/configs (excluding docs/plan.md)."""
        res = subprocess.run(
            ["git", "ls-files"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        tracked_files = [line.strip() for line in res.stdout.splitlines() if line.strip()]
        banned_patterns = [re.compile(r"\.venv-brain"), re.compile(r"python3\.12")]
        violations: list[str] = []

        for rel_path in tracked_files:
            if rel_path == "docs/plan.md":
                continue
            full_path = REPO_ROOT / rel_path
            if not full_path.is_file():
                continue
            try:
                content = full_path.read_text(encoding="utf-8", errors="ignore")
            except Exception:
                continue
            for pat in banned_patterns:
                if pat.search(content):
                    violations.append(f"{rel_path}: matches {pat.pattern}")

        self.assertEqual(violations, [], f"Disallowed references found: {violations}")

    def test_no_duplicate_top_level_keys_in_settings(self):
        """No duplicate top-level keys across config/settings.d/*.yaml (if directory exists)."""
        settings_dir = REPO_ROOT / "config" / "settings.d"
        if not settings_dir.is_dir():
            return

        seen_keys: dict[str, str] = {}
        duplicates: list[str] = []
        for yml_file in sorted(settings_dir.glob("*.yaml")):
            try:
                data = yaml.safe_load(yml_file.read_text(encoding="utf-8"))
            except Exception as exc:
                self.fail(f"Could not parse YAML {yml_file}: {exc}")
            if not isinstance(data, dict):
                continue
            for key in data.keys():
                if key in seen_keys:
                    duplicates.append(f"Key '{key}' in {yml_file.name} duplicates key in {seen_keys[key]}")
                else:
                    seen_keys[key] = yml_file.name

        self.assertEqual(duplicates, [])

    def test_lockfile_matches_uv_pip_freeze(self):
        """requirements.lock.txt matches uv pip freeze --python .venv/bin/python."""
        lock_path = REPO_ROOT / "requirements.lock.txt"
        self.assertTrue(lock_path.is_file(), "requirements.lock.txt missing")

        res = subprocess.run(
            ["uv", "pip", "freeze", "--python", ".venv/bin/python"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        frozen_lines = sorted(line.strip() for line in res.stdout.splitlines() if line.strip())
        lock_lines = sorted(line.strip() for line in lock_path.read_text().splitlines() if line.strip())

        self.assertEqual(frozen_lines, lock_lines, "requirements.lock.txt drift detected against uv pip freeze")

    def test_uv_pip_check_and_disallowed_packages(self):
        """uv pip check only reports allowed overrides and disallowed packages are absent."""
        res = subprocess.run(
            ["uv", "pip", "check", "--python", ".venv/bin/python"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
        )
        allowed_override_names = {"onnxruntime", "opencv-python", "tflite-runtime"}
        for line in (res.stdout + res.stderr).splitlines():
            line = line.strip()
            if not line or line.startswith("Checked ") or line.startswith("Found "):
                continue
            matches_allowed = any(name in line for name in allowed_override_names)
            self.assertTrue(
                matches_allowed,
                f"Unexpected incompatibility reported by uv pip check: {line}",
            )

        list_res = subprocess.run(
            ["uv", "pip", "list", "--python", ".venv/bin/python"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=True,
        )
        installed_names = {
            parts[0].lower() for parts in (line.split() for line in list_res.stdout.splitlines()) if parts
        }
        self.assertNotIn("onnxruntime", installed_names, "Plain CPU onnxruntime package must not be installed")
        self.assertNotIn(
            "opencv-python", installed_names, "Plain opencv-python must not be installed (use opencv-contrib-python)"
        )

    def test_infra_acl_matches_generator_if_present(self):
        """If tools/gen_acl.py and infra/acl exist, infra/acl matches tools/gen_acl.py output."""
        gen_script = REPO_ROOT / "tools" / "gen_acl.py"
        acl_file = REPO_ROOT / "infra" / "acl"
        if gen_script.is_file() and acl_file.is_file():
            res = subprocess.run(
                [".venv/bin/python", str(gen_script)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=True,
            )
            self.assertEqual(acl_file.read_text(encoding="utf-8").strip(), res.stdout.strip())

    def test_fake_llm_backend_scoping(self):
        """FakeLLMBackend is not imported in brain/ outside brain/llm.py."""
        brain_dir = REPO_ROOT / "brain"
        if not brain_dir.is_dir():
            return
        violations: list[str] = []
        for py_path in brain_dir.rglob("*.py"):
            if py_path.name == "llm.py":
                continue
            content = py_path.read_text(encoding="utf-8", errors="ignore")
            if "FakeLLMBackend" in content:
                violations.append(str(py_path.relative_to(REPO_ROOT)))
        self.assertEqual(violations, [], f"FakeLLMBackend imported outside brain/llm.py: {violations}")


if __name__ == "__main__":
    unittest.main()
