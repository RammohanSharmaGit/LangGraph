import asyncio
from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
from dotenv import load_dotenv
from custom_react import make_custom_react_agent

initial_message = """open a new tab and go to sive.rs url, 
click on the hyperlink : live
and take a screenshot"""


async def main():
    client = MultiServerMCPClient(
        {
            "playwright": {
                "url": "http://localhost:8931/sse",
                "transport": "sse",
            },
        }
    )

    async with client.session("playwright") as session:
        print("Session created with the tool server. The browser will remain open.")
        print("Type 'exit' or 'quit' to close.")

        tools = await load_mcp_tools(session)
        agent = make_custom_react_agent(tools=tools)
        loop = asyncio.get_running_loop()

        if initial_message:
            print(f"\nRunning initial prompt:\n---\n{initial_message}\n---")
            try:
                response = await agent.ainvoke({"messages": initial_message})
                print(response)
            except Exception as e:
                print(f"An error occurred during agent invocation: {e}")

        while True:
            try:
                prompt = await loop.run_in_executor(
                    None, input, "\nEnter your next prompt (or 'exit' to quit): "
                )
            except (KeyboardInterrupt, EOFError):
                break

            if prompt.lower() in ["exit", "quit"]:
                break

            if not prompt.strip():
                continue

            try:
                response = await agent.ainvoke({"messages": prompt})
                print(response)
            except Exception as e:
                print(f"An error occurred during agent invocation: {e}")

    print("\nSession closed.")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\nProgram interrupted by user.")