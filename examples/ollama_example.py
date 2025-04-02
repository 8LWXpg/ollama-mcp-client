import asyncio
import sys
from abstract.server_config import ConfigContainer
from clients.ollama_client import OllamaMCPClient


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_config.json>")
        sys.exit(1)

    config = ConfigContainer.form_file(sys.argv[1])
    async with await OllamaMCPClient.create(config) as client:
        print("client initiated")
        print("\nMCP Client Started!")
        print("Type your queries or 'quit' to exit.")

        while True:
            try:
                query = input("\nChat: ").strip()

                match query.lower():
                    case "quit":
                        break
                    case "clear":
                        await client.prepare_prompt()
                        continue

                async for part in client.process_message(query):
                    if part["role"] == "assistant":
                        message = part["content"]
                        print(message, end="", flush=True)

            except Exception as e:
                client.logger.error(e)


if __name__ == "__main__":
    asyncio.run(main())
