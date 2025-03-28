from abc import ABC, abstractmethod
from typing import Optional
from contextlib import AsyncExitStack
from mcp import ClientSession, StdioServerParameters

import logging
import colorlog


class AbstractMCPClient(ABC):
    """Abstract base class for MCP clients"""

    def __init__(self):
        self.session: Optional[ClientSession] = None
        self.exit_stack = AsyncExitStack()

        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(logging.DEBUG)

        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.DEBUG)
        formatter = colorlog.ColoredFormatter(
            "%(log_color)s%(levelname)s%(reset)s - %(message)s",
            datefmt=None,
            reset=True,
            log_colors={
                "DEBUG": "cyan",
                "INFO": "green",
                "WARNING": "yellow",
                "ERROR": "red",
                "CRITICAL": "bold_red",
            },
        )

        console_handler.setFormatter(formatter)
        if not self.logger.hasHandlers():
            self.logger.addHandler(console_handler)

    @abstractmethod
    async def connect_to_server(self, server_params: StdioServerParameters) -> None:
        """
        Connect to an MCP server

        Args:
            commandline: commandline to execute MCP server
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
