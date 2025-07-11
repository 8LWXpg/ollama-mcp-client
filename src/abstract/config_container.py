import json
from typing import Self, List, Optional
from mcp import StdioServerParameters
from pydantic import RootModel, BaseModel


class HttpServerConfig(BaseModel):
    """Configuration for HTTP servers"""
    url: str
    name: Optional[str] = None


class ConfigContainer(RootModel):
    """
    Root model to represent the entire JSON structure with dynamic key.
    """

    root: dict[str, StdioServerParameters]

    def __getitem__(self, index: int) -> tuple:
        if not self.root:
            raise ValueError("No configurations found")

        name = list(self.root.keys())[index]
        return name, self.root[name]

    def items(self):
        """Return only stdio server items, excluding http_servers"""
        return {k: v for k, v in self.root.items() if k != "http_servers"}.items()

    def get_http_servers(self) -> List[HttpServerConfig]:
        """Get all HTTP server configurations"""
        http_servers = []
        
        if "http_servers" in self.root:
            http_config = self.root["http_servers"]
            if isinstance(http_config, list):
                for i, server in enumerate(http_config):
                    if isinstance(server, str):
                        http_servers.append(HttpServerConfig(url=server, name=f"http_{i}"))
                    elif isinstance(server, dict):
                        http_servers.append(HttpServerConfig(**server))
        
        return http_servers

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
            stdio_data = {k: v for k, v in json_data.items() if k != "http_servers"}
            instance = cls(root=stdio_data)
            instance.root["http_servers"] = json_data.get("http_servers", [])
            return instance
        except Exception as e:
            raise ValueError(f"Error processing configuration: {e}")