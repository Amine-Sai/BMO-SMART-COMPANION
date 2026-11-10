from __future__ import annotations

import re
from typing import Iterable, Iterator
from brain.llm import ChatChunk
from brain.settings import AudioSettings

TITLES = set("mr. mrs. ms. dr. prof. sr. jr. st. mt. vs. fig. no. vol. e.g. i.e.".split())
MAYBE = set("etc. inc. ltd. co. u.s. u.s.a. u.k. a.i. ph.d. a.m. p.m.".split())
PUNCT_RE = re.compile(r"(\n+|(?:\.{3,}|…|[.!?]+)(?:['\"”’\)]*))")
NUMERIC_LIST_RE = re.compile(
    r"(?:^|[\n\r]|[:\-*#]\s*)\s*(?:(?:step|item|rule|part)\s+)?\d+(?:\.\d+)*\Z",
    re.IGNORECASE,
)


class SentenceSplitter:
    """Incrementally splits streamed tokens into complete sentences."""

    def __init__(self, max_clause_chars: int | None = None) -> None:
        if max_clause_chars is None:
            max_clause_chars = AudioSettings().tts_max_clause_chars
        self.max_clause_chars = max_clause_chars
        self._buf = ""

    def feed(self, chunk: str | ChatChunk) -> list[str]:
        text = chunk.text if isinstance(chunk, ChatChunk) else str(chunk)
        if not text:
            return []
        self._buf += text
        return list(self._extract())

    def flush(self) -> Iterator[str]:
        yield from self._extract()
        rem = self._buf.strip()
        self._buf = ""
        if rem:
            yield rem

    def _extract(self) -> Iterator[str]:
        while True:
            self._buf = self._buf.lstrip(" \t\r\n")
            if not self._buf:
                break
            search_pos, found = 0, False
            while True:
                m = PUNCT_RE.search(self._buf, search_pos)
                if not m:
                    break
                p_start, p_end, matched = m.start(), m.end(), m.group(0)
                if matched.startswith("\n"):
                    sent = self._buf[:p_start].strip()
                    self._buf = self._buf[p_end:]
                    if sent:
                        yield sent
                    found = True
                    break

                rest = self._buf[p_end:]
                if not rest:
                    break
                if rest[0] not in " \t\r\n":
                    search_pos = p_end
                    continue

                prefix = self._buf[:p_start]
                wm = re.search(r"([A-Za-z0-9.]+)\Z", prefix)
                word = (wm.group(1).lower() + ".") if wm else ""
                is_numeric_list = bool(NUMERIC_LIST_RE.search(prefix))
                if word in TITLES or re.match(r"^[a-z]\.$", word) or is_numeric_list:
                    search_pos = p_end
                    continue

                after_ws = rest.lstrip(" \t\r\n")
                is_ambig = (
                    matched.startswith("...")
                    or matched.startswith("…")
                    or word in MAYBE
                    or any(c in matched for c in "\"'”’)")
                )
                if is_ambig:
                    if not after_ws:
                        break
                    look = after_ws.lstrip("\"'“‘(")
                    if not look:
                        break
                    if look[0].islower():
                        search_pos = p_end
                        continue

                sent = self._buf[:p_end].strip()
                self._buf = self._buf[p_end:]
                if sent:
                    yield sent
                found = True
                break

            if found:
                continue

            if self.max_clause_chars > 0 and len(self._buf) > self.max_clause_chars:
                split_idx = self._find_clause_split()
                if split_idx is not None and split_idx > 0:
                    sent = self._buf[:split_idx].strip()
                    self._buf = self._buf[split_idx:]
                    if sent:
                        yield sent
                    continue
            break

    def _find_clause_split(self) -> int | None:
        window = self._buf[: self.max_clause_chars]
        clause_matches = list(re.finditer(r"[,;:]['\"”’\)]*", window))
        for cm in reversed(clause_matches):
            c_start = cm.start()
            if not any(c.isalnum() for c in window[: cm.end()]):
                continue
            ch = window[c_start]
            if ch in (",", ":"):
                prev_c = window[c_start - 1] if c_start > 0 else ""
                next_c = self._buf[c_start + 1] if c_start + 1 < len(self._buf) else ""
                if prev_c.isdigit() and next_c.isdigit():
                    continue
                if ch == ":" and window[:c_start].endswith(("http", "https")):
                    continue
            return cm.end()

        space_matches = list(re.finditer(r"\s+", window))
        if space_matches:
            s_idx = space_matches[-1].start()
            if any(c.isalnum() for c in window[:s_idx]):
                return s_idx

        rest = self._buf[self.max_clause_chars :]
        m_rest = re.search(r"[,;:\s]", rest)
        if m_rest:
            end_pos = self.max_clause_chars + m_rest.start()
            if m_rest.group(0) in ",;:":
                end_pos = self.max_clause_chars + m_rest.end()
            return end_pos
        return self.max_clause_chars


def split_sentences(
    chunks: Iterable[str | ChatChunk],
    max_clause_chars: int | None = None,
) -> Iterator[str]:
    """Yield complete sentences from an iterable of string or ChatChunk tokens."""
    splitter = SentenceSplitter(max_clause_chars=max_clause_chars)
    for chunk in chunks:
        yield from splitter.feed(chunk)
    yield from splitter.flush()
