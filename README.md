# MCP client using Ollama

This is a simple [MCP](https://modelcontextprotocol.io) client that uses the Ollama API to get the latest MCP data.

It follows the [MCP Client](https://modelcontextprotocol.io/tutorials/building-a-client) docs, but swaps out the
Claude specific code for Ollama.

Only supports stdio transport for now.

## Usage

Install the package in development mode:

```bash
pip install -e .
```

```python
uv run examples/ollama_example.py /path/to/server.py
```
