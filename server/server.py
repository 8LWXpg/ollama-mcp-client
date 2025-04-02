# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
#     "mcp[cli]",
# ]
# ///

import random
import httpx
from mcp.server.fastmcp import FastMCP
import math

# Initialize FastMCP server
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


if __name__ == "__main__":
    # Initialize and run the server
    mcp.run(transport="stdio")
