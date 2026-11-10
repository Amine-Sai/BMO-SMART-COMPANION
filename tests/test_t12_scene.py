from __future__ import annotations

import asyncio
import socket
import threading
import time
from bmo_contracts.models import LastSign, Scene, ScenePerson, VisionStatus
from bmo_contracts.topics import VISION_CONTEXT, VISION_STATUS
from brain.llm import count_tokens
from brain.scene import SceneCache
from brain.settings import SceneSettings, load_settings
from paho.mqtt.client import CallbackAPIVersion, Client
from tools.mock_cv import MockCV
from tools.simple_broker import SimpleMQTTBroker


def _get_free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def test_load_settings_scene() -> None:
    settings = load_settings()
    assert settings.scene is not None
    assert settings.scene.max_age_s == 10.0


def test_mock_cv_broker_integration() -> None:
    port = _get_free_port()
    broker = SimpleMQTTBroker(host="127.0.0.1", port=port, require_auth=False)
    loop = asyncio.new_event_loop()

    def run_broker() -> None:
        asyncio.set_event_loop(loop)
        loop.run_until_complete(broker.start())
        loop.run_forever()

    broker_thread = threading.Thread(target=run_broker, daemon=True)
    broker_thread.start()
    time.sleep(0.15)

    mock_cv = MockCV(host="127.0.0.1", port=port)
    mock_cv.start()
    time.sleep(0.15)

    sample_scene = Scene(
        ts=time.time(),
        people=[
            ScenePerson(
                track_id=None,
                name="Jud",
                pos="center",
                proximity="near",
                since_s=12.0,
            )
        ],
        objects=["cup", "book"],
        last_sign=LastSign(text="HELLO", age_s=2.0),
        recent=["Jud arrived"],
    )
    mock_cv.client.publish(VISION_CONTEXT, sample_scene.model_dump_json(), retain=True)
    time.sleep(0.1)

    settings = SceneSettings(max_age_s=10.0)
    cache = SceneCache(settings=settings)

    sub = Client(CallbackAPIVersion.VERSION2)
    sub.on_message = lambda _c, _u, msg: cache.handle_mqtt_message(msg.topic, msg.payload)
    sub.connect("127.0.0.1", port)
    sub.loop_start()
    sub.subscribe([(VISION_CONTEXT, 0), (VISION_STATUS, 0)])

    deadline = time.monotonic() + 3.0
    while time.monotonic() < deadline and (cache.current() is None or not cache.is_online()):
        time.sleep(0.05)

    try:
        current_scene = cache.current()
        assert current_scene is not None
        assert isinstance(current_scene, Scene)
        prompt_line = cache.as_prompt_line()
        assert prompt_line.startswith("Camera: online")
        assert (
            prompt_line
            == "Camera: online | people: Jud(center,near,12s) | objects: cup,book | sign: HELLO(2s ago) | recent: Jud arrived"
        )
        assert count_tokens(prompt_line) <= 60
    finally:
        sub.loop_stop()
        sub.disconnect()
        mock_cv.stop()
        loop.call_soon_threadsafe(loop.stop)


def test_scene_cache_staleness() -> None:
    fake_time = 100.0
    settings = SceneSettings(max_age_s=10.0)
    cache = SceneCache(settings=settings, clock=lambda: fake_time)

    sample_scene = Scene(
        ts=fake_time,
        people=[],
        objects=[],
        last_sign=None,
        recent=[],
    )
    cache.update_scene(sample_scene)
    cache.update_status(VisionStatus(online=True))

    assert cache.current() is not None
    assert cache.is_stale() is False

    fake_time += 10.5
    assert cache.is_stale() is True
    assert cache.current() is None
    assert cache.as_prompt_line() == "Camera: offline/stale"


def test_scene_cache_offline_status() -> None:
    settings = SceneSettings(max_age_s=10.0)
    cache = SceneCache(settings=settings)

    sample_scene = Scene(
        ts=time.monotonic(),
        people=[],
        objects=[],
        last_sign=None,
        recent=[],
    )
    cache.update_scene(sample_scene)
    cache.update_status(VisionStatus(online=True))

    assert cache.is_online() is True
    assert cache.current() is not None

    cache.update_status(VisionStatus(online=False))
    assert cache.is_online() is False
    assert cache.is_stale() is True
    assert cache.current() is None
    assert cache.as_prompt_line() == "Camera: offline/stale"


def test_scene_cache_concurrency() -> None:
    settings = SceneSettings(max_age_s=2.0)
    cache = SceneCache(settings=settings)
    num_threads = 10
    iterations = 50
    barrier = threading.Barrier(num_threads)
    errors: list[Exception] = []

    def worker(tid: int) -> None:
        try:
            barrier.wait()
            for i in range(iterations):
                if tid % 3 == 0:
                    scene = Scene(
                        ts=time.monotonic(),
                        people=[ScenePerson(name=f"User_{tid}_{i}", pos="center", proximity="near", since_s=1.0)],
                        objects=[f"obj_{i}"],
                        last_sign=LastSign(text="TEST", age_s=0.5),
                        recent=[f"event_{i}"],
                    )
                    cache.update_scene(scene)
                elif tid % 3 == 1:
                    online = (i % 2 == 0)
                    cache.update_status(VisionStatus(online=online))
                else:
                    cache.handle_mqtt_message(
                        VISION_CONTEXT,
                        Scene(ts=time.monotonic(), people=[], objects=[], recent=[]).model_dump_json(),
                    )

                _ = cache.is_online()
                _ = cache.is_stale()
                curr = cache.current()
                if curr is not None:
                    assert isinstance(curr, Scene)
                prompt = cache.as_prompt_line()
                assert isinstance(prompt, str)
                assert prompt.startswith("Camera: ")
        except Exception as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=worker, args=(i,)) for i in range(num_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5.0)

    assert not any(t.is_alive() for t in threads), "Worker threads hung (possible deadlock)"
    assert not errors, f"Errors encountered during concurrency: {errors}"
