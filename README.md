# MCP client using Ollama

This is a simple [MCP](https://modelcontextprotocol.io) client that uses the Ollama API to get the latest MCP data.

It follows the [MCP Client](https://modelcontextprotocol.io/tutorials/building-a-client) docs, but swaps out the
Claude specific code for Ollama.

Only supports stdio transport for now.

## Usage

Install the package in development mode:

```shell
git clone https://github.com/8LWXpg/ollama-mcp-client
cd ollama-mcp-client
uv venv
uv pip install -e .
```

Run the Streamable HTTP Server before running the client (The Streamable HTTP Server should run before the client running, or else the client will return error)
```shell
uv run httpserver/server.py
```

Run the Client
```shell
uv run examples/ollama_example.py examples/server.json
```
