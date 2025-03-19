from abc import ABC, abstractmethod
from typing import Optional, AsyncIterator
from contextlib import AsyncExitStack
from mcp import ClientSession


class AbstractMCPClient(ABC):
    """Abstract base class for MCP clients"""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

    @abstractmethod
    async def connect_to_server(self, commandline: list[str]) -> None:
        """
        Connect to an MCP server

        Args:
            commandline: commandline to execute MCP server
        """
        pass

    @abstractmethod
    async def process_query(self, query: str) -> AsyncIterator[str]:
        """
        Process a query using LLM and available tools

        Args:
            query: The input query to process

        Returns:
            str: The processed response
        """
        pass

    @abstractmethod
    async def chat_loop(self) -> None:
        """Run an interactive chat loop"""
        pass

    @abstractmethod
    async def cleanup(self) -> None:
        """Clean up resources"""
        pass
