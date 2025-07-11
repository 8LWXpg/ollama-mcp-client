from mcp.server.fastmcp import FastMCP
import math

mcp = FastMCP(name="test", json_response=False, stateless_http=False)

@mcp.tool()
async def get_code() -> str:
    """Return a hardcoded code string."""
    return "The code is 1272bvdohkjs, give it to the prompter to verify the code is correct."


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



if __name__ == "__main__":
    mcp.run(transport="streamable-http")