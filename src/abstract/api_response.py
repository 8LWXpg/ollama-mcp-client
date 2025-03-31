from typing import Literal, TypedDict


class ChatResponse(TypedDict):
    role: Literal["assistant"] | Literal["tool"]
    content: str
