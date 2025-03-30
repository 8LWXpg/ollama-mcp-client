import asyncio
import sys
from src.abstract.server_config import ConfigContainer
from src.clients.ollama_client import OllamaMCPClient


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_config.json>")
        sys.exit(1)

    config = ConfigContainer.form_file(sys.argv[1])
    async with await OllamaMCPClient("http://192.168.0.33:11434").create(config) as client:
        print("client initiated")
        await client.chat_loop()


if __name__ == "__main__":
    asyncio.run(main())
