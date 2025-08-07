from datetime import timedelta
import json
from typing import Any, Self

from mcp import StdioServerParameters
from pydantic import BaseModel, Field


class SSEParameters(BaseModel):
    url: str
    headers: dict[str, Any] | None = None
    # TODO: Add parameters


class StreamableParameters(BaseModel):
    url: str
    headers: dict[str, Any] | None = None
    # TODO: Add parameters


class ConfigContainer(BaseModel):
    """
    Root model to represent the entire JSON structure with dynamic key.
    """

    stdio: dict[str, StdioServerParameters] = Field(default_factory=dict)
    sse: dict[str, SSEParameters] = Field(default_factory=dict)
    streamable: dict[str, StreamableParameters] = Field(default_factory=dict)

    @classmethod
    def from_file(cls, file_path: str) -> Self:
        """Read config from file

        Args:
            file_path (str): Path to file

        Returns:
            Self: ConfigContainer
        """
        try:
            with open(file_path, "r") as file:
                json_content = file.read()
            return cls.model_validate_json(json_content)
        except Exception as e:
            raise ValueError(f"Error reading/parsing file: {e}")
