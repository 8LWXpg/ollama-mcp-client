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
from typing import cast
import math
import httpx
import os
from dotenv import load_dotenv

mcp = FastMCP("zabbix")

load_dotenv()

ZABBIX_API_URL   = cast(str, os.getenv("ZABBIX_API_URL"))
ZABBIX_AUTH_TOKEN = cast(str, os.getenv("ZABBIX_AUTH_TOKEN"))

@mcp.tool()
async def get_host_problems(hostname: str) -> list:
    """
    Get list of active problems (triggers) for a given Zabbix host name.

    Args:
        hostname (str): Hostname in Zabbix.

    Returns:
        list: List of active trigger descriptions, or empty list if none.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(ZABBIX_API_URL, json={
            "jsonrpc": "2.0",
            "method": "host.get",
            "params": {
                "output": ["hostid"],
                "filter": {"host": hostname}
            },
            "auth": ZABBIX_AUTH_TOKEN,
            "id": 1
        })
        resp.raise_for_status()
        hosts = resp.json().get("result", [])
        if not hosts:
            return [] 

        host_id = hosts[0]["hostid"]

        resp = await client.post(ZABBIX_API_URL, json={
            "jsonrpc": "2.0",
            "method": "trigger.get",
            "params": {
                "output": ["description", "priority", "value"],
                "hostids": [host_id],
                "filter": {"value": 1},
                "expandDescription": True,
                "only_true": True,
                "skipDependent": True,
                "sortfield": "priority",
                "sortorder": "DESC"
            },
            "auth": ZABBIX_AUTH_TOKEN,
            "id": 2
        })
        resp.raise_for_status()
        triggers = resp.json().get("result", [])

        return ["\n".join(f"- {t['description']}" for t in triggers)]
    
@mcp.tool()
async def get_all_problems() -> list:
    """
    Get all active problems across all hosts in Zabbix.

    Returns:
        list: Single-element list containing all problem descriptions formatted as a bullet list.
    """
    async with httpx.AsyncClient() as client:
        resp = await client.post(ZABBIX_API_URL, json={
            "jsonrpc": "2.0",
            "method": "trigger.get",
            "params": {
                "output": ["description", "priority", "value"],
                "filter": {
                    "value": 1 
                },
                "expandDescription": True,
                "only_true": True,
                "skipDependent": True,
                "sortfield": "priority",
                "sortorder": "DESC"
            },
            "auth": ZABBIX_AUTH_TOKEN,
            "id": 1
        })
        resp.raise_for_status()
        triggers = resp.json().get("result", [])

        return ["\n".join(f"- {t['description']}" for t in triggers)]


if __name__ == "__main__":
    mcp.run(transport="stdio")
    
