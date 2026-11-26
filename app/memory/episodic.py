from typing import TypedDict


class EpisodeInput(TypedDict):
    user_id: str
    thread_id: str
    task: str
    actions: list[str]
    result: str
    success: bool
