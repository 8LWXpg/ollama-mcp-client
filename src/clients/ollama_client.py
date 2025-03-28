from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from src.abstract.base_client import AbstractMCPClient

from typing import AsyncIterator, Sequence
from ollama import AsyncClient, Message, Tool

from src.abstract.server_config import ConfigContainer

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
        self.messages = []

    async def connect_to_multiple_servers(self, config: ConfigContainer):
        for name, params in config.items():
            session, tools = await self.connect_to_server(name, params)
            self.session[name] = session
            self.tools.extend(tools)

        self.logger.info(f"Connected to server with tools: {[tool['function']['name'] for tool in self.tools]}")

    async def connect_to_server(self, name: str, server_params: StdioServerParameters) -> tuple[ClientSession, Sequence[Tool]]:
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py)
        """
        stdio, write = await self.exit_stack.enter_async_context(stdio_client(server_params))
        session: ClientSession = await self.exit_stack.enter_async_context(ClientSession(stdio, write))

        await session.initialize()

        # List available tools
        response = await session.list_tools()
        tools = [
            Tool(
                type="function",
                function=Tool.Function(
                    name=f"{name}/{tool.name}",
                    description=tool.description,
                    parameters=tool.inputSchema,  # type: ignore
                ),
            )
            for tool in response.tools
        ]
        return (session, tools)

    async def prepare_prompt(self):
        """Clear current message and create new one"""
        default_prompts: list[str] = []
        for name, session in self.session.items():
            for prompt in (await session.list_prompts()).prompts:
                if prompt.name == "default":
                    default_prompts.append((await session.get_prompt(prompt.name)).messages[0].content.text)  # type: ignore
        # prompt = (await self.session.get_prompt("default")).messages  # type: ignore
        # pre_tool: Sequence[Message.ToolCall] = [Message.ToolCall(function=Message.ToolCall.Function(name="list_tables", arguments={}))]
        self.messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "system", "content": "\n".join(default_prompts)},
            # {
            #     "role": "tool",
            #     "content": (await self.tool_call(pre_tool))[0],
            # },
        ]

    async def process_query(self, query: str) -> AsyncIterator[str]:
        """Process a query using LLM and available tools"""
        self.messages.append({"role": "user", "content": query})

        async for part in self.recursive_prompt():
            yield part

    async def recursive_prompt(self) -> AsyncIterator[str]:
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
            split = tool.function.name.split("/")
            session = self.session[split[0]]
            tool_name = split[1]
            tool_args = tool.function.arguments

            # Execute tool call
            try:
                result = await session.call_tool(tool_name, dict(tool_args))
                self.logger.debug(f"Tool call result: {result.content}")
                message = f"tool: {tool.function}\nargs: {tool_args}\nreturn: {result.content[0].text}"  # type: ignore
            except Exception as e:
                self.logger.debug(f"Tool call error: {e}")
                message = f"Error in tool: {tool.function}\nargs: {tool_args}\n{e}"

            # Continue conversation with tool results
            messages.append(message)
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
