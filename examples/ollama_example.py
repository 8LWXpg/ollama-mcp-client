import asyncio
import sys
from src.abstract.server_config import ConfigContainer
from src.clients.ollama_client import OllamaMCPClient


async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_config.json>")
        sys.exit(1)

    client = OllamaMCPClient()
    print("client initiated")
    try:
        config = ConfigContainer.form_file(sys.argv[1])
        await client.connect_to_multiple_servers(config)
        await client.chat_loop()
    finally:
        await client.cleanup()


if __name__ == "__main__":
    asyncio.run(main())
