import asyncio
import re
import sys
from abstract.config_container import ConfigContainer
from clients.ollama_client import OllamaMCPClient

async def main():
    if len(sys.argv) < 2:
        print("Usage: python client.py <path_to_server_config.json> [http_url1] [http_url2] ...")
        print("       python client.py <http_url>")
        sys.exit(1)

    server_input = sys.argv[1]
    additional_http_urls = sys.argv[2:] if len(sys.argv) > 2 else []
    
    if server_input.startswith('http'):
        async with OllamaMCPClient() as client:
            await client.connect_to_streamable_http_server(server_input)
            
            for i, url in enumerate(additional_http_urls):
                if url.startswith('http'):
                    await client.connect_to_streamable_http_server(url, server_name=f"http_{i+1}")
            
            await client.prepare_prompt()
            await run_chat_loop(client)
    else:
        config = ConfigContainer.form_file(server_input)
        async with await OllamaMCPClient.create(config) as client:
            
            http_servers = config.get_http_servers()
            for http_server in http_servers:
                await client.connect_to_streamable_http_server(
                    http_server.url, 
                    server_name=http_server.name
                )
            
            for i, url in enumerate(additional_http_urls):
                if url.startswith('http'):
                    await client.connect_to_streamable_http_server(url, server_name=f"http_extra_{i}")
            
            await client.prepare_prompt()
            await run_chat_loop(client)


async def run_chat_loop(client: OllamaMCPClient):
    print("client initiated")
    print("\nMCP Client Started!")
    print("Type your queries or 'quit' to exit.")
    print("Commands: 'clear', 'server <name>', 'list_servers', 'list_tools'")

    while True:
        try:
            query = input("\nChat: ").strip()

            match query.lower():
                case "quit":
                    break
                case "clear":
                    await client.prepare_prompt()
                    continue
                case "list_servers":
                    print(f"Available servers: {list(client.servers.keys())}")
                    print(f"Selected servers: {list(client.selected_server.keys())}")
                    continue
                case "list_tools":
                    tools = client.get_tools()
                    continue
                case server_command if server_match := re.match(r"server (\w+)", server_command):
                    server_name = server_match.group(1)
                    if server_name in client.servers:
                        client.select_server([server_name])
                        print(f"Selected server: {server_name}")
                    else:
                        print(f"Server '{server_name}' not found. Available: {list(client.servers.keys())}")
                    continue
                case server_command if server_match := re.match(r"server (.+)", server_command):
                    server_names = [name.strip() for name in server_match.group(1).split(',')]
                    valid_servers = [name for name in server_names if name in client.servers]
                    if valid_servers:
                        client.select_server(valid_servers)
                        print(f"Selected servers: {valid_servers}")
                    else:
                        print(f"No valid servers found. Available: {list(client.servers.keys())}")
                    continue

            async for part in client.process_message(query):
                if part["role"] == "assistant":
                    message = part["content"]
                    print(message, end="", flush=True)

        except Exception as e:
            client.logger.error(e)


if __name__ == "__main__":
    asyncio.run(main())