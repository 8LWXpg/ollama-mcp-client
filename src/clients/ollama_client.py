from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ollama import AsyncClient
from src.abstract.base_client import AbstractMCPClient

from typing import Any, AsyncIterator, Sequence, Optional, Tuple
from ollama import Message

SYSTEM_PROMPT = """You are a helpful assistant capable of accessing external functions and engaging in casual chat. Use the responses from these function calls to provide accurate and informative answers. The answers should be natural and hide the fact that you are using tools to access real-time information. Guide the user about available tools and their capabilities. Always utilize tools to access real-time information when required. Engage in a friendly manner to enhance the chat experience.

# Notes

- Ensure responses are based on the latest information available from function calls.
- Maintain an engaging, supportive, and friendly tone throughout the dialogue.
- Always highlight the potential of available tools to assist users comprehensively."""


class OllamaMCPClient(AbstractMCPClient):
    def __init__(self):
        # Initialize session and client objects
        super().__init__()

        self.client = AsyncClient("http://192.168.0.33:11434")
        self.tools = []

    async def connect_to_server(self, commandline: list[str]):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py)
        """
        # if not commandline.endswith(".py"):
        #     raise ValueError("Server script must be a .py or .js file")

        server_params = StdioServerParameters(command=commandline[0], args=commandline[1:], env=None)

        stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
        self.stdio, self.write = stdio_transport
        self.session = await self.exit_stack.enter_async_context(ClientSession(self.stdio, self.write))

        await self.session.initialize()

        # List available tools
        response = await self.session.list_tools()
        self.tools = [
            {
                "type": "function",
                "function": {
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.inputSchema,
                },
            }
            for tool in response.tools
        ]
        print(
            "\nConnected to server with tools:",
            [tool["function"]["name"] for tool in self.tools],
        )

    async def process_query(self, query: str) -> AsyncIterator[str]:
        """Process a query using LLM and available tools"""
        messages: list[dict[str, Any]] = [
            # {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": query},
        ]

        async for part in self.recursive_prompt(messages):
            yield part

    async def recursive_prompt(self, messages: list[dict[str, Any]]) -> AsyncIterator[str]:
        stream = await self.client.chat(
            model="qwen2.5:7b",
            messages=messages,
            tools=self.tools,
            stream=True,
        )

        tool_calls: Sequence[Message.ToolCall] = []
        # content_buffer = []
        async for part in stream:
            if part.message.content:
                # content_buffer.append(part.message.content)
                yield part.message.content
            elif part.message.tool_calls:
                # TODO: call tools separately instead of all together
                tool_calls += part.message.tool_calls

        if len(tool_calls) > 0:
            tool_messages, tool_results = await self.tool_call(tool_calls)
            for tool_message in tool_messages:
                messages.append({"role": "user", "content": tool_message})
            async for part in self.recursive_prompt(messages):
                yield part

    async def tool_call(self, tool_calls: Sequence[Message.ToolCall]) -> Tuple[list[str], list[dict[str, Any]]]:
        messages: list[str] = []
        tool_results: list[dict] = []
        for tool in tool_calls:
            tool_name = tool.function.name
            tool_args = tool.function.arguments

            # Execute tool call
            result = await self.session.call_tool(tool_name, dict(tool_args))
            tool_results.append({"call": tool_name, "result": result})
            # print(tool_results)
            message = f"Calling tool {tool_name} with args {tool_args} returned {result.content[0].text}"
            print(f"[{message}]")

            # Continue conversation with tool results
            messages.append(message)
        return messages, tool_results

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "quit":
                    break

                async for response in self.process_query(query):
                    print(response, end="", flush=True)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
