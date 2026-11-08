from __future__ import annotations

import re
from typing import Iterable, Iterator
from brain.llm import ChatChunk

TITLES = set("mr. mrs. ms. dr. prof. sr. jr. st. mt. vs. fig. no. vol. e.g. i.e.".split())
MAYBE = set("etc. inc. ltd. co. u.s. u.s.a. u.k. a.i. ph.d. a.m. p.m.".split())
PUNCT_RE = re.compile(r"(\n+|(?:\.{3,}|…|[.!?]+)(?:['\"”’\)]*))")


class SentenceSplitter:
    """Incrementally splits streamed tokens into complete sentences."""

    def __init__(self) -> None:
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
                if word in TITLES or re.match(r"^[a-z]\.$", word):
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

            if not found:
                break


def split_sentences(chunks: Iterable[str | ChatChunk]) -> Iterator[str]:
    """Yield complete sentences from an iterable of string or ChatChunk tokens."""
    splitter = SentenceSplitter()
    for chunk in chunks:
        yield from splitter.feed(chunk)
    yield from splitter.flush()
