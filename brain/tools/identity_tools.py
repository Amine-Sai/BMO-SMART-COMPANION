from typing import Any

from brain.tools.registry import ToolRegistry, ToolSpec
from brain.tools.schemas import PersonNameArgs
from brain.vision_client import VisionClient


class IdentityTools:
    def __init__(
        self,
        vision_client: VisionClient,
        enroll_timeout_s: float = 15.0,
        forget_timeout_s: float = 3.0,
    ) -> None:
        self._vision_client = vision_client
        self._enroll_timeout_s = enroll_timeout_s
        self._forget_timeout_s = forget_timeout_s

    def remember_person(self, args: PersonNameArgs) -> dict[str, Any]:
        res = self._vision_client.enroll(args.name, timeout_s=self._enroll_timeout_s)
        return {"ok": res.ok, "id": res.id, "name": args.name, "detail": res.detail}

    def forget_person(self, args: PersonNameArgs) -> dict[str, Any]:
        res = self._vision_client.forget(args.name, timeout_s=self._forget_timeout_s)
        return {"ok": res.ok, "id": res.id, "name": args.name, "detail": res.detail}


def register_identity_tools(registry: ToolRegistry, vision_client: VisionClient) -> IdentityTools:
    tools = IdentityTools(vision_client=vision_client)
    registry.register(
        ToolSpec(
            name="remember_person",
            description="Enroll and remember a person's face.",
            schema=PersonNameArgs,
            handler=tools.remember_person,
            risky=False,
        )
    )
    registry.register(
        ToolSpec(
            name="forget_person",
            description="Forget an enrolled person's face.",
            schema=PersonNameArgs,
            handler=tools.forget_person,
            risky=True,
        )
    )
    return tools
