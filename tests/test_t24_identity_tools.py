from typing import Callable

from bmo_contracts.models import VisionCmd, VisionCmdResult
from bmo_contracts.topics import VISION_CMD
from brain.tools.identity_tools import IdentityTools, register_identity_tools
from brain.tools.registry import ToolRegistry
from brain.tools.schemas import PersonNameArgs
from brain.vision_client import VisionClient


class FakeBus:
    def __init__(self) -> None:
        self.published: list[tuple[str, bytes, int, bool]] = []
        self.published_cmds: list[VisionCmd] = []
        self.responder: Callable[[VisionCmd], VisionCmdResult | None] | None = None
        self.client: VisionClient | None = None

    def publish(self, topic: str, payload: bytes, qos: int = 0, retain: bool = False) -> None:
        self.published.append((topic, payload, qos, retain))
        if topic == VISION_CMD:
            cmd = VisionCmd.model_validate_json(payload)
            self.published_cmds.append(cmd)
            if self.responder and self.client:
                res = self.responder(cmd)
                if res is not None:
                    self.client.handle_cmd_result(res)


def test_remember_person_dispatch() -> None:
    registry = ToolRegistry()
    bus = FakeBus()
    client = VisionClient(bus.publish)
    bus.client = client
    register_identity_tools(registry, client)

    bus.responder = lambda cmd: VisionCmdResult(id=cmd.id, ok=True, detail="enrolled") if cmd.cmd == "enroll" else None

    res = registry.dispatch({"name": "remember_person", "arguments": {"name": "Jud"}})
    assert res["ok"] is True
    assert res["name"] == "Jud"
    assert res["detail"] == "enrolled"
    assert "id" in res

    assert len(bus.published) == 1
    assert len(bus.published_cmds) == 1
    cmd = bus.published_cmds[0]
    assert cmd.cmd == "enroll"
    assert cmd.args == {"name": "Jud", "shots": 8}


def test_forget_person_dispatch_confirmation() -> None:
    registry = ToolRegistry()
    bus = FakeBus()
    client = VisionClient(bus.publish)
    bus.client = client
    register_identity_tools(registry, client)

    bus.responder = lambda cmd: VisionCmdResult(id=cmd.id, ok=True, detail="forgotten") if cmd.cmd == "forget" else None

    # First call (unconfirmed): should return needs_confirmation and not publish
    res_unconfirmed = registry.dispatch(
        {"name": "forget_person", "arguments": {"name": "Jud"}},
        confirmed=False,
    )
    assert res_unconfirmed["ok"] is False
    assert res_unconfirmed["needs_confirmation"] is True
    assert res_unconfirmed["tool"] == "forget_person"
    assert len(bus.published) == 0
    assert len(bus.published_cmds) == 0

    # Second call (confirmed): should publish and return success
    res_confirmed = registry.dispatch(
        {"name": "forget_person", "arguments": {"name": "Jud"}},
        confirmed=True,
    )
    assert res_confirmed["ok"] is True
    assert res_confirmed["name"] == "Jud"
    assert res_confirmed["detail"] == "forgotten"
    assert "id" in res_confirmed

    assert len(bus.published) == 1
    assert len(bus.published_cmds) == 1
    cmd = bus.published_cmds[0]
    assert cmd.cmd == "forget"
    assert cmd.args == {"name": "Jud"}


def test_forget_person_failure_and_timeout() -> None:
    registry = ToolRegistry()
    bus = FakeBus()
    client = VisionClient(bus.publish)
    bus.client = client
    register_identity_tools(registry, client)

    bus.responder = lambda cmd: (
        VisionCmdResult(id=cmd.id, ok=False, detail="not enrolled") if cmd.cmd == "forget" else None
    )

    registry.dispatch(
        {"name": "forget_person", "arguments": {"name": "Unknown"}},
        confirmed=False,
    )
    res_fail = registry.dispatch(
        {"name": "forget_person", "arguments": {"name": "Unknown"}},
        confirmed=True,
    )
    assert res_fail["ok"] is False
    assert res_fail["name"] == "Unknown"
    assert res_fail["detail"] == "not enrolled"

    bus_timeout = FakeBus()
    client_timeout = VisionClient(bus_timeout.publish)
    fast_tools = IdentityTools(client_timeout, forget_timeout_s=0.01)
    res_timeout = fast_tools.forget_person(PersonNameArgs(name="Jud"))
    assert res_timeout["ok"] is False
    assert len(bus_timeout.published) == 1
