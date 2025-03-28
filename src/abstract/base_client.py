from abc import ABC
from contextlib import AsyncExitStack
from mcp import ClientSession

import logging
import colorlog


class AbstractMCPClient(ABC):
    """Abstract base class for MCP clients"""

    def __init__(self):
        self.session: dict[str, ClientSession] = {}
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
