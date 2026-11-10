"""Tests for streaming sentence splitter (T1.4)."""

from __future__ import annotations

import pytest
from brain.llm import ChatChunk
from brain.settings import AudioSettings
from brain.stream import SentenceSplitter, split_sentences

TRICKY_CASES: list[tuple[str, list[str]]] = [
    # 1. Honorific titles (Mr., Mrs., Ms.)
    (
        "Mr. Smith met Mrs. Jones and Ms. Davis yesterday. They discussed plans.",
        ["Mr. Smith met Mrs. Jones and Ms. Davis yesterday.", "They discussed plans."],
    ),
    # 2. Medical and academic titles (Dr., Prof.)
    (
        "Dr. Watson assisted Prof. Moriarty. The case was intriguing.",
        ["Dr. Watson assisted Prof. Moriarty.", "The case was intriguing."],
    ),
    # 3. Lineage suffixes (Sr., Jr.)
    (
        "John Smith Jr. visited Robert Brown Sr. at noon. It was pleasant.",
        ["John Smith Jr. visited Robert Brown Sr. at noon.", "It was pleasant."],
    ),
    # 4. Place and saint prefixes (St., Mt.)
    (
        "St. Patrick visited Mt. Everest on vacation. He took photos.",
        ["St. Patrick visited Mt. Everest on vacation.", "He took photos."],
    ),
    # 5. Versus abbreviation (vs.)
    (
        "Arsenal vs. Chelsea was a thrilling match. Both teams played well.",
        ["Arsenal vs. Chelsea was a thrilling match.", "Both teams played well."],
    ),
    # 6. Latin abbreviations (e.g.)
    (
        "Eat citrus fruits (e.g. lemons or oranges). That is healthy.",
        ["Eat citrus fruits (e.g. lemons or oranges).", "That is healthy."],
    ),
    # 7. Latin abbreviations (i.e.)
    (
        "Choose a primary color, i.e. red or blue. Do not pick yellow.",
        ["Choose a primary color, i.e. red or blue.", "Do not pick yellow."],
    ),
    # 8. Document reference abbreviations (Fig., No., Vol.)
    (
        "See Fig. 1 and Vol. 2 for No. 42. Details are provided there.",
        ["See Fig. 1 and Vol. 2 for No. 42.", "Details are provided there."],
    ),
    # 9. In-sentence etc.
    (
        "Apples, oranges, etc. are tasty fruits. We should buy more.",
        ["Apples, oranges, etc. are tasty fruits.", "We should buy more."],
    ),
    # 10. Sentence-ending etc.
    (
        "We bought apples, oranges, etc. Then we returned home.",
        ["We bought apples, oranges, etc.", "Then we returned home."],
    ),
    # 11. Corporate suffix in-sentence (Inc.)
    (
        "Acme Inc. announced new quarterly profits. Investors were pleased.",
        ["Acme Inc. announced new quarterly profits.", "Investors were pleased."],
    ),
    # 12. Corporate suffix at sentence boundary (Inc.)
    (
        "She works at Acme Inc. He works at Wayne Enterprises.",
        ["She works at Acme Inc.", "He works at Wayne Enterprises."],
    ),
    # 13. Corporate suffix (Ltd.)
    (
        "Widgets Ltd. agreed to terms. Work began immediately.",
        ["Widgets Ltd. agreed to terms.", "Work began immediately."],
    ),
    # 14. Corporate suffix (Co.)
    (
        "Smith & Co. was founded long ago. It still thrives today.",
        ["Smith & Co. was founded long ago.", "It still thrives today."],
    ),
    # 15. Acronym in-sentence (U.S.)
    (
        "The U.S. government met yesterday. New policies were enacted.",
        ["The U.S. government met yesterday.", "New policies were enacted."],
    ),
    # 16. Acronym (U.S.A.)
    (
        "The U.S.A. team took the gold medal. Fans cheered loudly.",
        ["The U.S.A. team took the gold medal.", "Fans cheered loudly."],
    ),
    # 17. Acronym (U.K.)
    (
        "They traveled to the U.K. on business. The weather was rainy.",
        ["They traveled to the U.K. on business.", "The weather was rainy."],
    ),
    # 18. Acronym (A.I.)
    (
        "Advances in A.I. are accelerating rapidly. Many researchers agree.",
        ["Advances in A.I. are accelerating rapidly.", "Many researchers agree."],
    ),
    # 19. Academic degree (Ph.D.)
    (
        "He earned his Ph.D. in computer science. Now he teaches courses.",
        ["He earned his Ph.D. in computer science.", "Now he teaches courses."],
    ),
    # 20. In-sentence time (a.m.)
    (
        "The meeting starts at 9 a.m. tomorrow. Do not be late.",
        ["The meeting starts at 9 a.m. tomorrow.", "Do not be late."],
    ),
    # 21. Sentence-ending time (a.m.)
    (
        "The meeting starts at 9 a.m. Please arrive five minutes early.",
        ["The meeting starts at 9 a.m.", "Please arrive five minutes early."],
    ),
    # 22. In-sentence time (p.m.)
    (
        "Dinner is served at 7 p.m. every evening. Everyone gathers together.",
        ["Dinner is served at 7 p.m. every evening.", "Everyone gathers together."],
    ),
    # 23. Sentence-ending time (p.m.)
    (
        "The store closes at 8 p.m. Tonight is an exception.",
        ["The store closes at 8 p.m.", "Tonight is an exception."],
    ),
    # 24. Middle initial
    (
        "George W. Bush gave a speech. The audience listened intently.",
        ["George W. Bush gave a speech.", "The audience listened intently."],
    ),
    # 25. Name initials
    (
        "J. K. Rowling wrote famous fantasy books. Millions read them.",
        ["J. K. Rowling wrote famous fantasy books.", "Millions read them."],
    ),
    # 26. Decimal number
    (
        "The mathematical constant pi is roughly 3.14159. Euler proved many formulas.",
        ["The mathematical constant pi is roughly 3.14159.", "Euler proved many formulas."],
    ),
    # 27. Currency
    (
        "The total bill was $19.99 plus tax. I paid with credit.",
        ["The total bill was $19.99 plus tax.", "I paid with credit."],
    ),
    # 28. Leading zero decimal
    (
        "A tiny error of 0.5 was detected. It was corrected quickly.",
        ["A tiny error of 0.5 was detected.", "It was corrected quickly."],
    ),
    # 29. Semantic versioning
    (
        "We successfully deployed v1.2.3 to production. Everything is stable.",
        ["We successfully deployed v1.2.3 to production.", "Everything is stable."],
    ),
    # 30. Unit suffixes
    (
        "The response timeout is set to 2.5s for now. It can be tuned later.",
        ["The response timeout is set to 2.5s for now.", "It can be tuned later."],
    ),
    # 31. Ellipsis continuation (lowercase)
    (
        "Wait... what did you say? Speak up.",
        ["Wait... what did you say?", "Speak up."],
    ),
    # 32. Ellipsis boundary (uppercase)
    (
        "Wait... What did you say? Speak up.",
        ["Wait...", "What did you say?", "Speak up."],
    ),
    # 33. Unicode ellipsis continuation (lowercase)
    (
        "Thinking… wondering about the answer. Maybe tomorrow.",
        ["Thinking… wondering about the answer.", "Maybe tomorrow."],
    ),
    # 34. Unicode ellipsis boundary (uppercase)
    (
        "Thinking… Wondering about the answer. Maybe tomorrow.",
        ["Thinking…", "Wondering about the answer.", "Maybe tomorrow."],
    ),
    # 35. Quoted speech followed by new sentence
    (
        'He exclaimed "Hello!" Then he walked away.',
        ['He exclaimed "Hello!"', "Then he walked away."],
    ),
    # 36. Quoted dialogue attribution
    (
        '"Hello!" she whispered softly.',
        ['"Hello!" she whispered softly.'],
    ),
    # 37. Quoted continuation and second dialogue
    (
        '"Wait," she said. "Don\'t leave yet!"',
        ['"Wait," she said.', '"Don\'t leave yet!"'],
    ),
    # 38. Parenthetical continuation
    (
        "He asked (is this right?) and kept walking.",
        ["He asked (is this right?) and kept walking."],
    ),
    # 39. Parenthetical boundary
    (
        "(Is this really right?) Yes, absolutely.",
        ["(Is this really right?)", "Yes, absolutely."],
    ),
    # 40. Multi-punctuation
    (
        "What?! Are you completely serious?!",
        ["What?!", "Are you completely serious?!"],
    ),
    # 41. Newline-separated lines without final periods
    (
        "First line\nSecond line\nThird line",
        ["First line", "Second line", "Third line"],
    ),
    # 42. Newline-separated sentences with periods
    (
        "First line.\nSecond line.\nThird line.",
        ["First line.", "Second line.", "Third line."],
    ),
]


@pytest.mark.parametrize(("text", "expected"), TRICKY_CASES)
@pytest.mark.parametrize("chunk_size", [1, 2, 3])
def test_tricky_sentence_splitting_chunked(text: str, expected: list[str], chunk_size: int) -> None:
    chunks = [text[i : i + chunk_size] for i in range(0, len(text), chunk_size)]
    result = list(split_sentences(chunks))
    assert result == expected


def test_streaming_eagerness() -> None:
    tokens_consumed = 0
    tokens = [
        "First",
        " sentence",
        ".",
        " ",
        "Second",
        " sentence",
        " is",
        " significantly",
        " longer",
        ".",
    ]

    def token_stream():
        nonlocal tokens_consumed
        for tok in tokens:
            tokens_consumed += 1
            yield tok

    emitted: list[str] = []
    consumed_at_first: int | None = None
    for sentence in split_sentences(token_stream()):
        if consumed_at_first is None:
            consumed_at_first = tokens_consumed
        emitted.append(sentence)

    assert consumed_at_first is not None
    # Proves first sentence emitted before the second sentence finishes yielding
    assert consumed_at_first <= 4
    assert tokens_consumed == len(tokens)
    assert emitted == ["First sentence.", "Second sentence is significantly longer."]


def test_chat_chunk_objects() -> None:
    chunks = [
        ChatChunk(text="The model replied. "),
        ChatChunk(text="Here is another answer."),
        ChatChunk(text="", done=True),
    ]
    sentences = list(split_sentences(chunks))
    assert sentences == ["The model replied.", "Here is another answer."]


def test_sentence_splitter_direct_feed_and_flush() -> None:
    splitter = SentenceSplitter()
    out1 = splitter.feed("Hello world! ")
    assert out1 == ["Hello world!"]
    out2 = splitter.feed("Remaining text with no punct")
    assert out2 == []
    out3 = list(splitter.flush())
    assert out3 == ["Remaining text with no punct"]


def test_empty_and_whitespace_chunks() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("") == []
    assert splitter.feed("   \n\t  ") == []
    assert splitter.feed(ChatChunk(text="")) == []
    assert list(splitter.flush()) == []


def test_numeric_list_not_split() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("1. Item") == []
    assert list(splitter.flush()) == ["1. Item"]

    assert list(split_sentences(["1. Item. ", "2. Another item."])) == [
        "1. Item.",
        "2. Another item.",
    ]
    assert list(split_sentences(["1. Item\n2. Another item"])) == [
        "1. Item",
        "2. Another item",
    ]
    assert list(split_sentences(["Step 1. Check this carefully."])) == [
        "Step 1. Check this carefully.",
    ]


def test_dr_smith_not_split() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("Dr. Smith visited yesterday. ") == [
        "Dr. Smith visited yesterday."
    ]
    tokens = ["Dr.", " Smith", " arrived.", " He", " spoke."]
    assert list(split_sentences(tokens)) == ["Dr. Smith arrived.", "He spoke."]


def test_decimals_not_oversplit() -> None:
    splitter = SentenceSplitter()
    assert splitter.feed("The value is 3.5 for now. ") == ["The value is 3.5 for now."]
    tokens = ["The multiplier is 3.", "5 times higher. Next sentence."]
    assert list(split_sentences(tokens)) == [
        "The multiplier is 3.5 times higher.",
        "Next sentence.",
    ]


def test_clause_split_delimiters_and_space() -> None:
    # Comma split
    s_comma = SentenceSplitter(max_clause_chars=50)
    out = s_comma.feed("This is a clause before comma, and this is another clause that exceeds limit")
    assert out == ["This is a clause before comma,"]
    assert list(s_comma.flush()) == ["and this is another clause that exceeds limit"]

    # Semicolon split
    s_semi = SentenceSplitter(max_clause_chars=50)
    out = s_semi.feed("This is a clause before semi; and this is another clause that exceeds limit")
    assert out == ["This is a clause before semi;"]
    assert list(s_semi.flush()) == ["and this is another clause that exceeds limit"]

    # Colon split
    s_colon = SentenceSplitter(max_clause_chars=50)
    out = s_colon.feed("Here is a header before colon: and this is another clause that exceeds limit")
    assert out == ["Here is a header before colon:"]
    assert list(s_colon.flush()) == ["and this is another clause that exceeds limit"]

    # Space split (no clause punctuation)
    s_space = SentenceSplitter(max_clause_chars=50)
    out = s_space.feed("This is a sentence with no punctuation whatsoever that exceeds the limit")
    assert out == ["This is a sentence with no punctuation whatsoever"]
    assert list(s_space.flush()) == ["that exceeds the limit"]


def test_clause_split_settings_default() -> None:
    splitter = SentenceSplitter()
    assert splitter.max_clause_chars == AudioSettings().tts_max_clause_chars
    assert splitter.max_clause_chars == 80


def test_clause_split_streaming_eagerness() -> None:
    splitter = SentenceSplitter(max_clause_chars=35)
    tokens = ["This ", "is ", "an ", "unpunctuated ", "stream ", "that ", "never ", "ends"]
    emitted: list[str] = []
    tokens_consumed_at_first: int | None = None
    for i, tok in enumerate(tokens):
        out = splitter.feed(tok)
        if out and tokens_consumed_at_first is None:
            tokens_consumed_at_first = i + 1
        emitted.extend(out)

    assert tokens_consumed_at_first is not None
    # Clause is emitted before all tokens finish yielding
    assert tokens_consumed_at_first < len(tokens)
    emitted.extend(splitter.flush())
    assert emitted == [
        "This is an unpunctuated stream",
        "that never ends",
    ]

