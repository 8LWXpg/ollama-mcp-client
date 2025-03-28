import json
from logging import root
from typing import Dict, Self
from mcp import StdioServerParameters
from pydantic import BaseModel, RootModel


class ConfigContainer(RootModel):
    """
    Root model to represent the entire JSON structure with dynamic key.
    """

    root: Dict[str, StdioServerParameters]

    def __getitem__(self, index: int) -> tuple:
        """
        Convenience method to extract the first (and typically only) configuration.

        Returns:
            tuple: (name, ServerConfig)
        """
        if not self.root:
            raise ValueError("No configurations found")

        name = list(self.root.keys())[index]
        return name, self.root[name]

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
            return cls(root=json_data)
        except Exception as e:
            raise ValueError(f"Error processing configuration: {e}")
