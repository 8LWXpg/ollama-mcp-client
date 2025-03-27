from os import name
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.abstract.base_client import AbstractMCPClient

from typing import AsyncIterator, Sequence
from ollama import AsyncClient, Message

SYSTEM_PROMPT = """You are a helpful assistant capable of accessing external functions and engaging in casual chat.
Use the responses from these function calls to provide accurate and informative answers.
The answers should be natural and hide the fact that you are using tools to access real-time information.
Guide the user about available tools and their capabilities.
Always utilize tools to access real-time information when required.
Engage in a friendly manner to enhance the chat experience.

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

        await self.session.initialize()  # type: ignore

        # List available tools
        response = await self.session.list_tools()  # type: ignore
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
        self.logger.info(f"Connected to server with tools: {[tool['function']['name'] for tool in self.tools]}")
        await self.prepare_prompt()

    async def prepare_prompt(self):
        """Predefined messages"""
        prompt = (await self.session.get_prompt("default")).messages  # type: ignore
        pre_tool: Sequence[Message.ToolCall] = [Message.ToolCall(function=Message.ToolCall.Function(name="list_tables", arguments={}))]
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": prompt[0].content.text},  # type: ignore
            {
                "role": "tool",
                "content": (await self.tool_call(pre_tool))[0],
            },
        ]

    async def process_query(self, query: str) -> AsyncIterator[str]:
        """Process a query using LLM and available tools"""
        self.messages.append({"role": "user", "content": query})

        async for part in self.recursive_prompt():
            yield part

    async def recursive_prompt(self) -> AsyncIterator[str]:
        # self.messages.extend(messages)
        # self.logger.debug(f"message: {self.messages}")
        # Streaming does not work when provided with tools, that's the issue with API or ollama itself.
        self.logger.debug("Prompting")
        stream = await self.client.chat(
            model="qwen2.5:14b",
            messages=self.messages,
            tools=self.tools,
            stream=True,
        )

        tool_messages: list[str] = []
        async for part in stream:
            if part.message.content:
                yield part.message.content
            elif part.message.tool_calls:
                self.logger.debug(f"Calling tool: {part.message.tool_calls}")
                tool_messages.extend(await self.tool_call(part.message.tool_calls))

        if len(tool_messages) > 0:
            for tool_message in tool_messages:
                self.messages.append({"role": "tool", "content": tool_message})
            async for part in self.recursive_prompt():
                yield part

    async def tool_call(self, tool_calls: Sequence[Message.ToolCall]) -> list[str]:
        messages: list[str] = []
        for tool in tool_calls:
            tool_name = tool.function.name
            tool_args = tool.function.arguments

            # Execute tool call
            result = await self.session.call_tool(tool_name, dict(tool_args))  # type: ignore
            self.logger.debug(f"Tool call result: {result.content}")
            message = f"tool: {tool_name}\nargs: {tool_args}\nreturn: {result.content[0].text}"  # type: ignore

            # Continue conversation with tool results
            messages.append(message)  # type: ignore
        return messages

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nChat: ").strip()

                match query.lower():
                    case "quit":
                        break
                    case "clear":
                        await self.prepare_prompt()
                        continue

                async for part in self.process_query(query):
                    print(part, end="", flush=True)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
