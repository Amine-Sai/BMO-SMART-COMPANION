#!/usr/bin/env python3
"""Interactive BMO MQTT Client, Live Monitor, and Command Verification Tool.

Usage:
  # Monitor all vision events, responses, and context feeds
  python tools/mqtt_client.py --monitor

  # Query pipeline status
  python tools/mqtt_client.py --cmd status

  # Fetch short-term visual memory context snapshot
  python tools/mqtt_client.py --cmd get_context

  # List registered biometric persons and custom objects
  python tools/mqtt_client.py --cmd list_persons
  python tools/mqtt_client.py --cmd list_objects

  # Enroll a person or object
  python tools/mqtt_client.py --cmd enroll_person --name Alice --shots 8
  python tools/mqtt_client.py --cmd register_object --name lighter --shots 6

  # Switch mode
  python tools/mqtt_client.py --cmd set_mode --mode lazy

  # Run automated test sweep
  python tools/mqtt_client.py --test-all

  # Interactive prompt menu
  python tools/mqtt_client.py --interactive
"""

from __future__ import annotations

import argparse
import json
import os
import queue
import sys
import time
from typing import Any, Dict, Optional

from bmo_contracts.topics import (
    VISION_CMD,
    VISION_CMD_RESULT,
    VISION_CONTEXT,
    VISION_EVENTS,
    VISION_WILDCARD,
)

try:
    import paho.mqtt.client as mqtt
    from paho.mqtt.enums import CallbackAPIVersion
except ImportError:
    print("[ERROR] paho-mqtt is required. Run in virtual environment: pip install paho-mqtt", file=sys.stderr)
    sys.exit(1)

# ANSI Color formatting
C_RESET = "\033[0m"
C_BOLD = "\033[1m"
C_RED = "\033[91m"
C_GREEN = "\033[92m"
C_YELLOW = "\033[93m"
C_BLUE = "\033[94m"
C_MAGENTA = "\033[95m"
C_CYAN = "\033[96m"
C_GRAY = "\033[90m"


class BMOMQTTTester:
    def __init__(self, host: str = "localhost", port: int = 1883):
        self.host = host
        self.port = int(port)
        self.client_id = f"bmo_tester_{int(time.time() * 1000) % 100000}"

        if hasattr(CallbackAPIVersion, "VERSION2"):
            self.client = mqtt.Client(CallbackAPIVersion.VERSION2, client_id=self.client_id)
        else:
            self.client = mqtt.Client(client_id=self.client_id)

        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.client.on_message = self._on_message

        self.connected = False
        self.response_queues: Dict[str, queue.Queue[Dict[str, Any]]] = {}
        self.all_responses: queue.Queue[Dict[str, Any]] = queue.Queue()
        self.monitor_mode = False

    def _on_connect(self, client: Any, userdata: Any, flags: Any, rc: Any, properties: Any = None) -> None:
        rc_val = getattr(rc, "value", rc)
        if rc_val == 0:
            self.connected = True
            client.subscribe(VISION_WILDCARD, qos=0)
            if not self.monitor_mode:
                # Silent connect unless in monitor mode
                pass
            else:
                print(
                    f"{C_GREEN}[CONNECTED]{C_RESET} Connected to broker {self.host}:{self.port}, listening on '{VISION_WILDCARD}'\n"
                )
        else:
            print(f"{C_RED}[ERROR]{C_RESET} Failed to connect to broker (rc: {rc_val})", file=sys.stderr)

    def _on_disconnect(
        self, client: Any, userdata: Any, flags_or_rc: Any, rc: Any = None, properties: Any = None
    ) -> None:
        self.connected = False

    def _on_message(self, client: Any, userdata: Any, msg: Any) -> None:
        topic = msg.topic
        payload_raw = msg.payload.decode("utf-8", errors="replace")
        try:
            data = json.loads(payload_raw)
        except Exception:
            data = {"raw": payload_raw}

        # Check for command response correlation (only from response topic)
        req_id = data.get("request_id") or data.get("id")
        if topic == VISION_CMD_RESULT and req_id and req_id in self.response_queues:
            self.response_queues[req_id].put(data)
        self.all_responses.put(data)

        if self.monitor_mode:
            self._print_monitor_message(topic, data)

    def _print_monitor_message(self, topic: str, data: Dict[str, Any]) -> None:
        now_str = time.strftime("%H:%M:%S")

        if topic == VISION_EVENTS:
            ev_type = data.get("event", "unknown")
            ev_data = data.get("data", {})
            print(
                f"{C_GRAY}[{now_str}]{C_RESET} {C_CYAN}{C_BOLD}[EVENT]{C_RESET} {C_BOLD}{ev_type.upper()}{C_RESET}: {json.dumps(ev_data)}"
            )

        elif topic == VISION_CMD_RESULT:
            cmd = data.get("command", "")
            status = data.get("status", "")
            req_id = data.get("request_id", "")
            status_color = (
                C_GREEN
                if status in ("ok", "success", "completed", "applied")
                else (C_YELLOW if status == "started" else C_RED)
            )
            print(
                f"{C_GRAY}[{now_str}]{C_RESET} {C_MAGENTA}{C_BOLD}[RESPONSE]{C_RESET} {cmd} -> {status_color}{status.upper()}{C_RESET} {C_GRAY}(req: {req_id}){C_RESET}"
            )
            if "data" in data and data["data"]:
                print(f"         {C_GRAY}{json.dumps(data['data'])}{C_RESET}")

        elif topic == VISION_CONTEXT:
            ctx = data.get("context", {})
            summary = ctx.get("summary", "")
            req_id = data.get("request_id")
            tag = f" {C_GRAY}(req: {req_id}){C_RESET}" if req_id else ""
            print(f"{C_GRAY}[{now_str}]{C_RESET} {C_YELLOW}{C_BOLD}[CONTEXT]{C_RESET}{tag} {summary}")

        elif topic == VISION_CMD:
            cmd = data.get("command", "")
            print(
                f"{C_GRAY}[{now_str}]{C_RESET} {C_BLUE}{C_BOLD}[COMMAND SENT]{C_RESET} {cmd} {C_GRAY}{json.dumps(data)}{C_RESET}"
            )

        else:
            print(f"{C_GRAY}[{now_str}]{C_RESET} [{topic}] {json.dumps(data)}")

    def start(self) -> bool:
        try:
            self.client.connect(self.host, self.port, keepalive=30)
            self.client.loop_start()
            # Wait up to 2 seconds for connection
            t0 = time.time()
            while not self.connected and (time.time() - t0) < 2.5:
                time.sleep(0.05)
            if not self.connected:
                print(
                    f"{C_RED}[ERROR]{C_RESET} Could not connect to MQTT broker at {self.host}:{self.port}.",
                    file=sys.stderr,
                )
                print("       Make sure a broker is running, e.g.: python tools/simple_broker.py", file=sys.stderr)
                return False
            return True
        except Exception as exc:
            print(f"{C_RED}[ERROR]{C_RESET} Connection failed: {exc}", file=sys.stderr)
            print("       Start broker first: python tools/simple_broker.py", file=sys.stderr)
            return False

    def stop(self) -> None:
        try:
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass

    def send_command(
        self,
        command: str,
        params: Optional[Dict[str, Any]] = None,
        timeout: float = 6.0,
        wait_for_response: bool = True,
    ) -> Optional[Dict[str, Any]]:
        """Publish a command and wait for correlated response."""
        req_id = f"req-{command}-{int(time.time() * 1000) % 10000}"
        payload: Dict[str, Any] = {"command": command, "request_id": req_id}
        if params:
            payload.update(params)

        resp_q: queue.Queue[Dict[str, Any]] = queue.Queue()
        self.response_queues[req_id] = resp_q

        topic_cmd = os.environ.get("BMO_MQTT_TOPIC_CMD", VISION_CMD)
        t0 = time.perf_counter()
        self.client.publish(topic_cmd, json.dumps(payload), qos=0)

        if not wait_for_response:
            return None

        try:
            resp = resp_q.get(timeout=timeout)
            lat_ms = (time.perf_counter() - t0) * 1000.0
            resp["_latency_ms"] = round(lat_ms, 1)
            return resp
        except queue.Empty:
            return None
        finally:
            if req_id in self.response_queues:
                del self.response_queues[req_id]


# ---------------------------------------------------------------------------
# CLI Modes
# ---------------------------------------------------------------------------


def run_monitor(tester: BMOMQTTTester) -> None:
    tester.monitor_mode = True
    print("\n====================================================================")
    print("  BMO Vision Real-Time MQTT Monitor")
    print(f"  Listening on: {tester.host}:{tester.port} ('{VISION_WILDCARD}')")
    print("  Press Ctrl+C to exit")
    print("====================================================================\n")
    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


def run_single_command(tester: BMOMQTTTester, cmd: str, args: argparse.Namespace) -> None:
    params: Dict[str, Any] = {}

    if args.json:
        try:
            params = json.loads(args.json)
            if "command" in params:
                cmd = params["command"]
        except Exception as exc:
            print(f"{C_RED}[ERROR]{C_RESET} Invalid JSON: {exc}", file=sys.stderr)
            return

    # Extract command-specific flags
    if args.name:
        params["name"] = args.name
    if args.shots is not None:
        params["shots"] = args.shots
    if args.synonyms:
        params["synonyms"] = [s.strip() for s in args.synonyms.split(",") if s.strip()]
    if args.old_name:
        params["old_name"] = args.old_name
    if args.new_name:
        params["new_name"] = args.new_name
    if args.mode:
        params["mode"] = args.mode
    if args.classes:
        params["classes"] = [c.strip() for c in args.classes.split(",") if c.strip()]
    if args.track_id is not None:
        params["track_id"] = args.track_id

    # Enrollment commands may take longer than default 6s
    timeout = float(args.timeout) if args.timeout else (25.0 if "enroll" in cmd or "register" in cmd else 6.0)

    print(f"{C_BOLD}[SENDING]{C_RESET} command='{cmd}' params={json.dumps(params)} (timeout={timeout}s)...")
    resp = tester.send_command(cmd, params, timeout=timeout)

    if resp is None:
        print(f"{C_RED}[TIMEOUT]{C_RESET} No response received within {timeout}s.")
        print("          Verify assistant_cv/main.py is running with MQTT enabled.")
    else:
        status = resp.get("status", "unknown")
        color = (
            C_GREEN
            if status in ("ok", "success", "completed", "applied")
            else (C_YELLOW if status == "started" else C_RED)
        )
        lat = resp.get("_latency_ms", 0.0)
        print(f"{color}{C_BOLD}[STATUS: {status.upper()}]{C_RESET} in {lat} ms")
        print(json.dumps(resp, indent=2))


def run_test_suite(tester: BMOMQTTTester) -> None:
    print("\n====================================================================")
    print("  BMO Vision Pipeline MQTT Automated Command Sweep")
    print(f"  Target Broker: {tester.host}:{tester.port}")
    print("====================================================================\n")

    tests = [
        ("Pipeline Status", "status", {}),
        ("Short-Term Context", "get_context", {}),
        ("List Biometric Persons", "list_persons", {}),
        ("List Custom Objects", "list_objects", {}),
        ("Set Lazy Mode", "set_mode", {"mode": "lazy"}),
        ("Set Active Mode", "set_mode", {"mode": "active"}),
        ("Set Lazy Mode (Restore)", "set_mode", {"mode": "lazy"}),
    ]

    passed = 0
    for name, cmd, params in tests:
        print(f"Checking '{name}' ({cmd})...", end=" ", flush=True)
        resp = tester.send_command(cmd, params, timeout=5.0)
        if resp is not None and resp.get("status") in ("ok", "success", "applied"):
            lat = resp.get("_latency_ms", 0.0)
            print(f"{C_GREEN}PASS{C_RESET} ({lat} ms)")
            passed += 1
        else:
            err = resp.get("status") if resp else "TIMEOUT"
            print(f"{C_RED}FAIL ({err}){C_RESET}")

    print(f"\nSweep Result: {passed}/{len(tests)} commands validated successfully.")


def run_interactive(tester: BMOMQTTTester) -> None:
    print("\n====================================================================")
    print("  BMO Vision Interactive MQTT Console")
    print("====================================================================")
    while True:
        print("\nChoose an action:")
        print("  1. Query Status")
        print("  2. Fetch Context Snapshot (LLM summary)")
        print("  3. List Biometric Identities")
        print("  4. List Enrolled Objects")
        print("  5. Enroll Person (interactive name)")
        print("  6. Register Custom Object (interactive name & shots)")
        print("  7. Delete Person")
        print("  8. Delete Object")
        print("  9. Toggle Lazy Mode (lazy / active)")
        print("  0. Exit")
        choice = input("\nSelect [0-9]: ").strip()

        if choice == "1":
            resp = tester.send_command("status")
            print(json.dumps(resp, indent=2))
        elif choice == "2":
            resp = tester.send_command("get_context")
            if resp and "data" in resp:
                print(f"\n{C_YELLOW}Scene Summary:{C_RESET} {resp['data'].get('context', {}).get('summary')}")
                print(json.dumps(resp["data"], indent=2))
        elif choice == "3":
            resp = tester.send_command("list_persons")
            print(json.dumps(resp, indent=2))
        elif choice == "4":
            resp = tester.send_command("list_objects")
            print(json.dumps(resp, indent=2))
        elif choice == "5":
            name = input("Enter person display name: ").strip() or "User"
            shots = int(input("Enter number of shots [8]: ").strip() or "8")
            print(f"Initiating enroll for '{name}'...")
            resp = tester.send_command("enroll_person", {"name": name, "shots": shots}, timeout=25.0)
            print(json.dumps(resp, indent=2))
        elif choice == "6":
            name = input("Enter object name (e.g. lighter): ").strip()
            if not name:
                continue
            shots = int(
                input("Enter shots (0 for instant zero-shot text, >0 for visual exemplar) [6]: ").strip() or "6"
            )
            syns = input("Enter comma-separated synonyms (optional): ").strip()
            syn_list = [s.strip() for s in syns.split(",") if s.strip()]
            resp = tester.send_command(
                "register_object", {"name": name, "shots": shots, "synonyms": syn_list}, timeout=25.0
            )
            print(json.dumps(resp, indent=2))
        elif choice == "7":
            name = input("Enter person name to delete: ").strip()
            if name:
                resp = tester.send_command("delete_person", {"name": name})
                print(json.dumps(resp, indent=2))
        elif choice == "8":
            name = input("Enter object name to delete: ").strip()
            if name:
                resp = tester.send_command("delete_object", {"name": name})
                print(json.dumps(resp, indent=2))
        elif choice == "9":
            mode = input("Enter mode (lazy / active / idle): ").strip() or "lazy"
            resp = tester.send_command("set_mode", {"mode": mode})
            print(json.dumps(resp, indent=2))
        elif choice == "0":
            break


def main() -> None:
    parser = argparse.ArgumentParser(description="BMO Assistant MQTT Inspection & Command Tool")
    parser.add_argument("-H", "--host", default=os.environ.get("BMO_MQTT_HOST", "localhost"), help="MQTT broker host")
    parser.add_argument(
        "-p", "--port", type=int, default=int(os.environ.get("BMO_MQTT_PORT", 1883)), help="MQTT broker port"
    )
    parser.add_argument("-m", "--monitor", action="store_true", help="Live stream and monitor vision topics")
    parser.add_argument("-c", "--cmd", help="Command name to execute")
    parser.add_argument("-n", "--name", help="Target person or object name")
    parser.add_argument("-s", "--shots", type=int, help="Target shots for enrollment")
    parser.add_argument("--synonyms", help="Comma-separated synonyms for object registration")
    parser.add_argument("--old-name", help="Old name for renaming")
    parser.add_argument("--new-name", help="New name for renaming")
    parser.add_argument("--mode", help="Mode for set_mode (lazy, active, idle)")
    parser.add_argument("--classes", help="Comma-separated class names for set_object_classes")
    parser.add_argument("--track-id", type=int, help="Track ID for recheck_face")
    parser.add_argument("-j", "--json", help="Raw JSON payload for command")
    parser.add_argument("-t", "--timeout", type=float, default=6.0, help="Response timeout in seconds")
    parser.add_argument("--test-all", action="store_true", help="Execute automated test sweep")
    parser.add_argument("-i", "--interactive", action="store_true", help="Launch interactive menu console")

    args = parser.parse_args()

    tester = BMOMQTTTester(host=args.host, port=args.port)
    if not tester.start():
        sys.exit(1)

    try:
        if args.monitor:
            run_monitor(tester)
        elif args.test_all:
            run_test_suite(tester)
        elif args.interactive:
            run_interactive(tester)
        elif args.cmd:
            run_single_command(tester, args.cmd, args)
        else:
            parser.print_help()
    finally:
        tester.stop()


if __name__ == "__main__":
    main()
