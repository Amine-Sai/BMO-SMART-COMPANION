"""Standalone BMO face display driven by the brain's shared state file.

The brain writes ``data/face_state.json`` with the mode, emotion, and an RMS
speech envelope. Polling that file is more reliable than MQTT for lip sync;
MQTT remains a fallback for runners that do not write it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import random
import re
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import cv2 as cv
import paho.mqtt.client as mqtt
from paho.mqtt.client import CallbackAPIVersion, MQTTMessage

from bmo_contracts.models import BrainSay, BrainStatus, Emotion
from bmo_contracts.topics import BRAIN_SAY, BRAIN_STATUS

from .assets import FaceArt, load_art
from .face import FaceParams, lerp_params, render_face, target_params
from .signal import FaceSignalReader, default_path, speech_mouth
from .state import FaceMode, FaceState


def speaking_mouth_open(t: float) -> bool:
    """Alternate the talking face while BMO speaks."""
    return math.sin(t * 18.0) + 0.4 * math.sin(t * 11.0) > 0.0


def _display_size() -> tuple[int, int]:
    """Best-effort native resolution of the active monitor."""
    try:
        result = subprocess.run(
            ["hyprctl", "monitors", "-j"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        if result.returncode == 0:
            monitors = json.loads(result.stdout)
            ordered = sorted(
                monitors,
                key=lambda m: (not m.get("focused"), m.get("id", 0)),
            )
            for monitor in ordered:
                if monitor.get("disabled"):
                    continue
                width = int(monitor.get("width", 0))
                height = int(monitor.get("height", 0))
                if width > 0 and height > 0:
                    return width, height
    except Exception:
        pass
    try:
        result = subprocess.run(
            ["xrandr", "--current"],
            capture_output=True,
            text=True,
            timeout=2,
        )
        match = re.search(r"current (\d+) x (\d+)", result.stdout)
        if match:
            return int(match.group(1)), int(match.group(2))
    except Exception:
        pass
    return 1280, 720


def _claim_instance() -> Path:
    """Stop a leftover face window so only the newest one is visible."""
    path = Path(
        os.environ.get(
            "BMO_FACE_PID",
            str(Path(__file__).resolve().parent.parent / "data" / "face.pid"),
        )
    )
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        old_pid = int(path.read_text(encoding="utf-8").strip())
    except (OSError, ValueError):
        old_pid = 0
    if old_pid > 0 and old_pid != os.getpid():
        cmdline = Path(f"/proc/{old_pid}/cmdline")
        try:
            is_face = "face_ui.main" in cmdline.read_bytes().decode(
                "utf-8", "replace"
            )
        except OSError:
            is_face = False
        if is_face:
            try:
                os.kill(old_pid, signal.SIGTERM)
                for _ in range(20):
                    time.sleep(0.05)
                    try:
                        os.kill(old_pid, 0)
                    except ProcessLookupError:
                        break
                else:
                    os.kill(old_pid, signal.SIGKILL)
            except OSError:
                pass
    try:
        path.write_text(str(os.getpid()), encoding="utf-8")
    except OSError:
        pass
    return path


class FaceApp:
    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args
        self.size = args.size or _display_size()
        self.state = FaceState(
            sleep_after_s=args.sleep_after,
            last_activity=time.monotonic(),
            mode=FaceMode.SLEEP,
        )
        self.events: queue.Queue[tuple[str, object]] = queue.Queue(maxsize=128)
        self.current = FaceParams()
        self.art: FaceArt | None = None
        self._next_blink = time.monotonic() + 3.0
        self._blink_until = 0.0
        self._client: mqtt.Client | None = None
        self._stop = threading.Event()
        self.signal = FaceSignalReader(
            Path(args.state) if getattr(args, "state", None) else None
        )
        # The face must always boot asleep. Ignore a state file left behind by
        # a previous run (old mtime) and retained MQTT statuses until the brain
        # has published its initial idle status.
        self._boot_wall = time.time()
        self._boot_ready = False
        self._signal: dict | None = None
        self._signal_seq: int | None = None

    # -- MQTT ---------------------------------------------------------------
    def _on_connect(
        self, client: mqtt.Client, userdata, flags, reason_code, properties
    ) -> None:
        if reason_code != 0:
            print(f"[FACE] MQTT connect failed: {reason_code}", flush=True)
            return
        print(
            f"[FACE] connected to {self.args.mqtt_host}:{self.args.mqtt_port}",
            flush=True,
        )
        client.subscribe(BRAIN_STATUS, qos=1)
        client.subscribe(BRAIN_SAY, qos=1)

    def _on_disconnect(self, client, userdata, flags, reason_code, properties) -> None:
        print(f"[FACE] MQTT disconnected: {reason_code}", flush=True)

    def _on_message(self, client, userdata, msg: MQTTMessage) -> None:
        try:
            if msg.topic == BRAIN_STATUS:
                status = BrainStatus.model_validate_json(msg.payload)
                event: tuple[str, object] = ("status", status)
            elif msg.topic == BRAIN_SAY:
                say = BrainSay.model_validate_json(msg.payload)
                event = ("say", say)
            else:
                return
        except Exception as exc:  # noqa: BLE001
            print(f"[FACE] bad payload on {msg.topic}: {exc}", flush=True)
            return
        try:
            self.events.put_nowait(event)
        except queue.Full:
            try:
                self.events.get_nowait()
            except queue.Empty:
                pass
            try:
                self.events.put_nowait(event)
            except queue.Full:
                pass

    def _start_mqtt(self) -> None:
        client = mqtt.Client(CallbackAPIVersion.VERSION2)
        client.on_connect = self._on_connect
        client.on_disconnect = self._on_disconnect
        client.on_message = self._on_message
        client.reconnect_delay_set(1, 10)
        client.connect_async(
            self.args.mqtt_host, self.args.mqtt_port, keepalive=30
        )
        self._client = client
        thread = threading.Thread(
            target=client.loop_forever,
            kwargs={"retry_first_connection": True},
            daemon=True,
            name="face-mqtt",
        )
        thread.start()

    def _apply_signal(self, now: float) -> None:
        try:
            mtime = self.signal.path.stat().st_mtime
        except OSError:
            return
        # A file written before this process started belongs to the previous
        # run, so it must not wake the freshly booted sleeping face.
        if mtime < self._boot_wall - 0.05:
            return
        signal = self.signal.read()
        if signal is None:
            return
        self._signal = signal
        self._boot_ready = True
        seq = signal.get("seq")
        if seq == self._signal_seq:
            return
        self._signal_seq = seq
        try:
            emotion = Emotion(str(signal.get("emotion", "neutral")))
        except ValueError:
            emotion = Emotion.NEUTRAL
        previous = self.state.mode
        self.state.emotion = emotion
        text = str(signal.get("text", ""))
        if text:
            self.state.last_say_text = text
        self.state.on_status(
            str(signal.get("mode", "idle")),
            bool(signal.get("online", True)),
            now,
        )
        if self.state.mode != previous:
            print(
                f"[FACE] file status={self.state.mode.value} "
                f"emotion={emotion.value}",
                flush=True,
            )
        if self.state.mode == FaceMode.SPEAKING:
            self.state.last_say = now

    def _drain(self, now: float) -> None:
        # The shared state file is authoritative when present; MQTT is only a
        # fallback for older runners that do not write it.
        if self._signal is not None:
            while True:
                try:
                    self.events.get_nowait()
                except queue.Empty:
                    return
        while True:
            try:
                kind, payload = self.events.get_nowait()
            except queue.Empty:
                return
            if kind == "status" and isinstance(payload, BrainStatus):
                if not self._boot_ready:
                    if str(payload.state) != "idle":
                        continue
                    self._boot_ready = True
                print(
                    f"[FACE] status={payload.state} online={payload.online}",
                    flush=True,
                )
                self.state.on_status(payload.state, payload.online, now)
            elif kind == "say" and isinstance(payload, BrainSay):
                if not self._boot_ready:
                    continue
                emotion = payload.emotion
                if not isinstance(emotion, Emotion):
                    try:
                        emotion = Emotion(str(emotion))
                    except ValueError:
                        emotion = Emotion.NEUTRAL
                print(
                    f"[FACE] say[{emotion.value}]: {payload.text[:80]!r}",
                    flush=True,
                )
                self.state.on_say(emotion, payload.text, now)

    # -- Rendering ----------------------------------------------------------
    def _blink_active(self, now: float) -> bool:
        if self.state.mode not in (
            FaceMode.AWAKE,
            FaceMode.LISTENING,
            FaceMode.SPEAKING,
        ):
            return False
        if now >= self._next_blink:
            self._blink_until = now + 0.12
            self._next_blink = now + random.uniform(2.4, 6.0)
        return now < self._blink_until

    def run(self) -> int:
        args = self.args
        if args.display:
            os.environ["DISPLAY"] = args.display
        if not args.no_art:
            self.art = load_art(self.size)
        cv.namedWindow(
            args.title,
            cv.WINDOW_FULLSCREEN if args.fullscreen else cv.WINDOW_NORMAL,
        )
        if args.fullscreen:
            cv.setWindowProperty(
                args.title, cv.WND_PROP_FULLSCREEN, cv.WINDOW_FULLSCREEN
            )
        else:
            cv.resizeWindow(args.title, self.size[0], self.size[1])

        self._start_mqtt()
        print(
            f"[FACE] state={self.signal.path}",
            flush=True,
        )
        print(
            f"[FACE] window {self.size[0]}x{self.size[1]} "
            f"fullscreen={args.fullscreen} display={os.environ.get('DISPLAY', '-')}",
            flush=True,
        )
        print(f"[FACE] sleep_after={self.state.sleep_after_s:.1f}s idle", flush=True)

        last = time.monotonic()
        try:
            while not self._stop.is_set():
                now = time.monotonic()
                dt = max(1e-3, min(0.1, now - last))
                last = now
                self._apply_signal(now)
                self._drain(now)
                previous_mode = self.state.mode
                self.state.tick(now)
                if self.state.mode != previous_mode:
                    print(
                        f"[FACE] state={self.state.mode.value} "
                        f"after {now - self.state.last_activity:.1f}s idle",
                        flush=True,
                    )
                blink = self._blink_active(now)
                if self.art is not None:
                    mouth_open = False
                    if self.state.mode == FaceMode.SPEAKING:
                        level = speech_mouth(self._signal, now)
                        mouth_open = (
                            speaking_mouth_open(now) if level is None else level > 0.12
                        )
                    frame = self.art.render(
                        self.state.mode,
                        self.state.emotion,
                        now,
                        blinking=blink,
                        speaking_open=mouth_open,
                    )
                else:
                    target = target_params(
                        self.state.mode,
                        self.state.emotion,
                        now,
                        self.state.last_say,
                        now,
                    )
                    if blink and target.eye_style == "normal" and not target.wink:
                        target.eye_open = 0.04
                    alpha = min(1.0, dt * 11.0)
                    self.current = lerp_params(self.current, target, alpha)
                    frame = render_face(
                        self.current, self.state.mode, now, self.size
                    )
                cv.imshow(args.title, frame)
                key = cv.waitKey(max(1, int(1000 / max(1, args.fps)))) & 0xFF
                if key in (27, ord("q")) and not args.no_quit:
                    break
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()
        return 0

    def stop(self) -> None:
        self._stop.set()
        if self._client is not None:
            try:
                self._client.disconnect()
            except Exception:
                pass
        try:
            cv.destroyAllWindows()
        except Exception:
            pass


def _parse_size(text: str) -> tuple[int, int]:
    try:
        w, h = text.lower().split("x", 1)
        width, height = int(w), int(h)
    except (ValueError, AttributeError) as exc:
        raise argparse.ArgumentTypeError(f"invalid size {text!r}") from exc
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError(f"invalid size {text!r}")
    return width, height


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="BMO face display")
    parser.add_argument("--mqtt-host", default="127.0.0.1", help="MQTT broker host")
    parser.add_argument("--mqtt-port", type=int, default=1883, help="MQTT broker port")
    parser.add_argument(
        "--size",
        type=_parse_size,
        default=None,
        help="face size, e.g. 1280x720 (default: native monitor resolution)",
    )
    parser.add_argument("--fullscreen", action="store_true", help="cover the screen")
    parser.add_argument("--display", default=None, help="X11 display, e.g. :0")
    parser.add_argument(
        "--no-art",
        action="store_true",
        help="use the procedural face instead of the SVG artwork",
    )
    parser.add_argument(
        "--sleep-after",
        type=float,
        default=60.0,
        help="seconds idle before the face falls asleep (0 disables)",
    )
    parser.add_argument("--fps", type=int, default=60, help="target frame rate")
    parser.add_argument(
        "--state",
        type=Path,
        default=default_path(),
        help="shared face state file written by the brain",
    )
    parser.add_argument("--title", default="BMO Face", help="window title")
    parser.add_argument(
        "--no-quit", action="store_true", help="ignore Esc/q (kiosk mode)"
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    pid_path = _claim_instance()
    app = FaceApp(args)
    try:
        return app.run()
    except Exception as exc:  # noqa: BLE001
        print(f"[FACE] fatal: {exc}", file=sys.stderr, flush=True)
        app.stop()
        return 1
    finally:
        try:
            if pid_path.read_text(encoding="utf-8").strip() == str(os.getpid()):
                pid_path.unlink()
        except OSError:
            pass


if __name__ == "__main__":
    raise SystemExit(main())
