from mcp import ClientSession
from ollama import Tool
from pydantic import BaseModel, ConfigDict


class Session(BaseModel):
    session: ClientSession
    tools: list[Tool]

    model_config = ConfigDict(arbitrary_types_allowed=True)
