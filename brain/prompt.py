from brain.llm import count_tokens
from brain.scene import SceneCache
from brain.settings import LLMSettings

_LLM_DEFAULTS = LLMSettings()

BMO_SYSTEM_PROMPT = (
    "You are BMO, a warm and curious robot friend. "
    "Always speak in one or two short spoken sentences. "
    "Only discuss the camera or what you can see when the user explicitly asks; "
    "otherwise never mention seeing, the camera, or the room. "
    "When a Scene line is present, rely strictly on it and never invent people, objects, or signs. "
    "If the Scene line says no fresh view, briefly say you could not get a clear look. "
    "Always ask for explicit confirmation before calling forget_person. "
    "Speak in plain English only, without lists or special formatting."
)


def _count_message_tokens(messages: list[dict[str, str]]) -> int:
    return sum(count_tokens(m["content"]) + 4 for m in messages)


class RollingHistory:
    def __init__(
        self,
        max_turns: int = _LLM_DEFAULTS.history_turns,
        max_total_tokens: int = _LLM_DEFAULTS.history_max_tokens,
    ) -> None:
        self.max_turns = max_turns
        self.max_total_tokens = max_total_tokens
        self._messages: list[dict[str, str]] = []

    @property
    def messages(self) -> list[dict[str, str]]:
        return [dict(m) for m in self._messages]

    def add_message(self, role: str, content: str) -> None:
        self._messages.append({"role": role, "content": content})
        max_messages = self.max_turns * 2
        if len(self._messages) > max_messages:
            self._messages = self._messages[-max_messages:]
        while self._messages and _count_message_tokens(self._messages) > self.max_total_tokens:
            self._messages.pop(0)

    def add_turn(self, user_text: str, assistant_text: str) -> None:
        self.add_message("user", user_text)
        self.add_message("assistant", assistant_text)

    def clear(self) -> None:
        self._messages.clear()


class PromptBuilder:
    def __init__(
        self,
        scene_cache: SceneCache,
        system_prompt: str = BMO_SYSTEM_PROMPT,
        max_turns: int = _LLM_DEFAULTS.history_turns,
        max_total_tokens: int = _LLM_DEFAULTS.history_max_tokens,
    ) -> None:
        self.scene_cache = scene_cache
        self.system_prompt = system_prompt
        self.max_turns = max_turns
        self.max_total_tokens = max_total_tokens
        self.history = RollingHistory(max_turns=max_turns, max_total_tokens=max_total_tokens)

    def record_turn(self, user_text: str, assistant_text: str) -> None:
        self.history.add_turn(user_text, assistant_text)

    def build_messages(self, user_text: str, scene_target: str | None = None) -> list[dict[str, str]]:
        content = self.system_prompt
        if scene_target is not None:
            content = f"{content}\nScene: {self.scene_cache.as_prompt_line(target=scene_target)}"
        system_msg = {"role": "system", "content": content}
        user_msg = {"role": "user", "content": user_text}
        history_msgs = self.history.messages
        messages = [system_msg, *history_msgs, user_msg]

        while history_msgs and _count_message_tokens(messages) > self.max_total_tokens:
            history_msgs.pop(0)
            messages = [system_msg, *history_msgs, user_msg]

        while _count_message_tokens(messages) > self.max_total_tokens and user_msg["content"]:
            user_msg["content"] = user_msg["content"][:-5]
            messages = [system_msg, *history_msgs, user_msg]

        return messages
