# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "httpx",
#     "mcp[cli]",
# ]
# ///

import httpx
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("rag")

# Before trying to use this tool, make sure you have a local RAG server running on port 8001.
@mcp.tool()
async def local_rag_query(query: str) -> str:
    try:
        async with httpx.AsyncClient(timeout=120.0) as client: 
            res = await client.post(
                "http://localhost:8001/query/local",
                json={"query": query},
            )
            res.raise_for_status()
            data = res.json()
            return data.get("result", "No result found")
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        return f"Error executing tool local_rag_query: {e}\nTraceback:\n{tb}"

if __name__ == "__main__":
    mcp.run(transport="stdio")
