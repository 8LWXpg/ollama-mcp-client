# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
#     "mcp[cli]",
# ]
# ///

import random
import string
import httpx
from mcp.server.fastmcp import FastMCP
import math

mcp = FastMCP("test")

@mcp.tool()
async def get_random() -> float:
    """Gets a truly random number (for real) (trust me)

    Returns:
        float: really random number
    """
    return random.Random().random()


@mcp.tool()
async def pow(a: float, b: float) -> float:
    """Calculate a of the power b

    Args:
        a (float): number
        b (float): power

    Returns:
        float: Calculated result
    """
    return math.pow(a, b)


@mcp.tool()
async def random_user(count: int) -> dict:
    """Generates random user data

    Args:
        count (int): returned user count

    Returns:
        dict: user data json as dict
    """
    res = httpx.get(f"https://randomuser.me/api?results={count}&inc=gender,name,email,phone,id")
    res.raise_for_status()
    return res.json()

@mcp.tool()
async def prompter_name() -> str:
    return "My name is Ryan"

# For testing purposes, returns the interface of local stdio MCP Server.
import subprocess
import re
from typing import Dict
@mcp.tool()
async def get_ip_interfaces() -> Dict[str, str]:
    """
    Returns a dictionary of network interfaces and their IP addresses on Ubuntu.
    Example: {"eth0": "192.168.1.10", "lo": "127.0.0.1"}
    """
    result = subprocess.run(["ip", "addr"], stdout=subprocess.PIPE, text=True)
    output = result.stdout

    interfaces = {}
    current_iface = None

    for line in output.splitlines():
        if line.startswith(" "):
            if current_iface and "inet " in line:
                match = re.search(r"inet (\d+\.\d+\.\d+\.\d+)", line)
                if match:
                    interfaces[current_iface] = match.group(1)
        else:
            match = re.match(r"\d+: ([\w\d@]+):", line)
            if match:
                current_iface = match.group(1).split("@")[0]

    return interfaces


if __name__ == "__main__":
    mcp.run(transport="stdio")
