import asyncio

from langchain_mcp_adapters.client import MultiServerMCPClient, load_mcp_tools
from langgraph.prebuilt import create_react_agent
from langchain_openai import ChatOpenAI
import os, sys
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model = "gpt-5-mini", api_key= key)

message = """open a new tab and go to sive.rs url, 
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

        tools = await load_mcp_tools(session)
        agent = create_react_agent(model=llm, tools=tools)

        response = await agent.ainvoke({"messages": message})
        print(response)

            # Now, wait for user input before exiting the 'async with' block.
            # As long as the script is paused here, the session is active.
        print("\nAgent action complete. The browser tab is persistent.")
        print("Press Enter in this terminal to close the session and exit.")
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, sys.stdin.readline)

    # This line will only be reached after you press Enter
    print("Session closed.")



if __name__ == "__main__":
    asyncio.run(main())