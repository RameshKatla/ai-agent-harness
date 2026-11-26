import re
import unicodedata
from collections.abc import Collection
from typing import TypedDict

from app.agent.state import Observation, ToolCall


class Trajectory(TypedDict):
    status: str
    final_answer: str | None
    error: str | None
    current_step: int
    tool_calls: list[ToolCall]
    observations: list[Observation]
    retrieved_memories: list[str]


_WORD = re.compile(r"(?<!\w)[+-]\d+(?:[.,]\d+)*|\w+(?:[.,]\d+)*")


def tokens(value: str, *, case_sensitive: bool = False) -> tuple[str, ...]:
    normalized = unicodedata.normalize("NFKC", value).replace("−", "-")
    return tuple(_WORD.findall(normalized if case_sensitive else normalized.casefold()))


def contains_phrase(text: str, phrase: str, *, case_sensitive: bool = False) -> bool:
    words = tokens(text, case_sensitive=case_sensitive)
    expected = tokens(phrase, case_sensitive=case_sensitive)
    if not expected:
        return phrase in text if case_sensitive else phrase.casefold() in text.casefold()
    size = len(expected)
    return any(words[index:index + size] == expected for index in range(len(words) - size + 1))


def fraction(matches: Collection[bool]) -> float:
    return sum(matches) / len(matches) if matches else 1.0


def retry_count(trajectory: Trajectory) -> int:
    return sum(max(0, len(observation["attempts"]) - 1) for observation in trajectory["observations"])


def execution_errors(trajectory: Trajectory) -> list[str]:
    errors = [trajectory["error"]] if trajectory["error"] else []
    for observation in trajectory["observations"]:
        attempts = observation["attempts"]
        errors.extend(f'{observation["tool"]}: {attempt["error"]}' for attempt in attempts if attempt["error"])
        if not attempts and not observation["success"]:
            errors.append(f'{observation["tool"]}: {observation["result"]}')
    return errors
