import json
from typing import Any, Self
from mcp import StdioServerParameters
from pydantic import BaseModel


class SSEParameters(BaseModel):
    url: str
    headers: dict[str, Any] | None = None


class ConfigContainer(BaseModel):
    """
    Root model to represent the entire JSON structure with dynamic key.
    """

    stdio: dict[str, StdioServerParameters]
    sse: dict[str, SSEParameters]

    @classmethod
    def form_file(cls, file_path: str) -> Self:
        """Read config from file

        Args:
            file_path (str): Path to file

        Returns:
            Self: ConfigContainer
        """
        try:
            with open(file_path, "r") as file:
                json_data = json.load(file)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            raise ValueError(f"Error reading file: {e}")

        try:
            return cls(**json_data)
        except Exception as e:
            raise ValueError(f"Error processing configuration: {e}")
