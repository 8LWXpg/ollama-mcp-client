from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from ollama import AsyncClient
from src.abstract.base_client import AbstractMCPClient

from typing import Any, AsyncIterator, Sequence
from ollama import Message

SYSTEM_PROMPT = """You are a helpful assistant capable of accessing external functions and engaging in casual chat. Use the responses from these function calls to provide accurate and informative answers. The answers should be natural and hide the fact that you are using tools to access real-time information. Guide the user about available tools and their capabilities. Always utilize tools to access real-time information when required. Engage in a friendly manner to enhance the chat experience.

# Tools
 
{tools}

# Notes:
  
- Ensure responses are based on the latest information available from function calls.
- Maintain an engaging, supportive, and friendly tone throughout the dialogue.
- Always highlight the potential of available tools to assist users comprehensively."""


class OllamaMCPClient(AbstractMCPClient):
    def __init__(self):
        # Initialize session and client objects
        super().__init__()

        self.DefaultModel = "qwen2.5:7b"
        self.client = AsyncClient("http://192.168.0.33:11434")
        self.tools = []
        self.model = None  # Initialize model as None

    async def connect_to_server(self, commandline: list[str]):
        """Connect to an MCP server

        Args:
            server_script_path: Path to the server script (.py)
        """
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

    async def get_available_models(self):
        """Get list of available models from Ollama server"""
        models_response = await self.client.list()
        # print(models_response["models"])
        return [model["model"] for model in models_response["models"]]

    async def select_model(self):
        """Let user select a model from available models"""
        print("\nAvailable models:")
        try:
            models = await self.get_available_models()
            print(f"\nType 'quit' or 'q' to use default model '{self.DefaultModel}'.")
            for i, model in enumerate(models, 1):
                if model == self.DefaultModel:
                    print(f"{i}. {model} (default)")
                else:
                    print(f"{i}. {model}")

            while True:
                try:
                    choice = input("\nSelect a model (number): ")
                    if (cjpi := choice.lower()) == "quit" or cjpi == "q":
                        print(f"Auto using default model: {self.DefaultModel}")
                        self.model = self.DefaultModel
                        break
                    index = int(choice) - 1
                    if 0 <= index < len(models):
                        self.model = models[index]
                        print(f"\nSelected model: {self.model}")
                        break
                    else:
                        print("Invalid selection. Please try again.")
                except ValueError:
                    print("Please enter a valid number.")
        except Exception as e:
            print(f"Error listing models: {str(e)}")
            print(f"Using default model: {self.DefaultModel}")
            self.model = self.DefaultModel

    async def process_query(self, query: str) -> AsyncIterator[str]:
        """Process a query using LLM and available tools"""
        messages: list[dict[str, Any]] = [
            # {
            #     "role": "system",
            #     "content": SYSTEM_PROMPT.format(tools="\n- ".join([f"{t['function']['name']}: {t['function']['description']}" for t in self.tools])),
            # },
            {"role": "user", "content": query},
        ]

        async for part in self.recursive_prompt(messages):
            yield part

    async def recursive_prompt(self, messages: list[dict[str, Any]]) -> AsyncIterator[str]:
        # Streaming does not work when providing with tools, that's the issue with API itself.
        stream = await self.client.chat(
            model=self.model if self.model else self.DefaultModel,
            messages=messages,
            tools=self.tools,
            stream=True,
        )
        tool_messages: list[dict[str, Any]] = []
        assistant_content = ""

        async for part in stream:
            if part.message.content:
                assistant_content += part.message.content
                yield part.message.content
            elif part.message.tool_calls:
                self.logger.debug(f"Calling tool: {part.message.tool_calls}")
                tool_results = await self.tool_call(part.message.tool_calls)
                tool_messages.extend(tool_results)

        # Add assistant's response to message history
        if assistant_content:
            messages.append({"role": "assistant", "content": assistant_content})

        # If tools were called, continue the conversation with tool results
        if len(tool_messages) > 0:
            for tool_message in tool_messages:
                messages.append(tool_message)

            # Get a new response that incorporates the tool results
            follow_up_messages = messages.copy()
            async for part in self.recursive_prompt(follow_up_messages):
                yield part

    async def tool_call(self, tool_calls: Sequence[Message.ToolCall]) -> list[dict[str, Any]]:
        messages: list[dict[str, Any]] = []
        for tool in tool_calls:
            tool_name = tool.function.name
            tool_args = tool.function.arguments

            # Execute tool call
            result = await self.session.call_tool(tool_name, dict(tool_args))  # type: ignore
            self.logger.debug(f"Tool call result: {result}")
            message = f"Called tool \033[1m {tool_name} \033[0;0m with args {tool_args}, return result: \033[1m {result.content[0].text} \033[0;0m"  # type: ignore
            self.logger.info(message)
            # Continue conversation with tool results
            messages.append(
                {
                    "role": "tool",
                    "name": tool_name,
                    "content": message,
                }
            )
        return messages

    async def chat_loop(self):
        """Run an interactive chat loop"""
        print("\nMCP Client Started!")

        # Let user select model before starting chat
        await self.select_model()

        print("\nType your queries or 'quit' to exit, 'select model' or 'sm' to change model.")

        while True:
            try:
                query = input("\nQuery: ").strip()

                if query.lower() == "sm" or query.lower() == "select model":
                    await self.select_model()
                    continue

                if query.lower() == "quit":
                    break

                async for part in self.process_query(query):
                    print(part, end="", flush=True)

            except Exception as e:
                print(f"\nError: {str(e)}")

    async def cleanup(self):
        """Clean up resources"""
        await self.exit_stack.aclose()
