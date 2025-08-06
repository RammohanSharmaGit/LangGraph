from typing import TypedDict, List, Sequence, Annotated
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage, BaseMessage, SystemMessage
from langgraph.graph import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool
import os

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash",
    google_api_key = key
)
#llm = ChatOpenAI(model = "gpt-4.1-mini-2025-04-14", api_key= key)

document_content = ""

class State(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def update_document(content : str) -> str:
    """this tool updates the document with provided content"""
    global document_content
    document_content = content
    return f"Updated document successfully. The current content is:\n{document_content}"

@tool
def save_document(filename : str) -> str:
    """this tool saves the document to a text file and finish the process
    Args:
        filename : name for the file
    """
    global document_content

    if not filename.endswith(".txt"):
        filename = f"{filename}.txt"

    try:
        with open(filename,"w") as file:
            file.write(document_content)
        print(f"\nDocument has been saved with name {filename}")
        return f"Document has been saved with name {filename}"
    except Exception as e:
        return f"Error occured : {str(e)}"
    
tools = [update_document, save_document]

llm.bind_tools(tools)

def agent(state : State) -> State:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. You are going to help the user update and modify documents.
    
    - If the user wants to update or modify content, use the 'update' tool with the complete updated content.
    - If the user wants to save and finish, you need to use the 'save' tool.
    - Make sure to always show the current document state after modifications.
    
    The current document content is:{document_content}
    """)

    if not state.get("messages"):
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)

    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\nUSER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state.get("messages",[])) + [user_message]
                   
    response = llm.invoke(all_messages)
    print(f"\n AI: {response.content} \n API USAGE : {response.usage_metadata}")

    if hasattr(response, "tool_calls") and response.tool_calls:
        print(f"USING TOOLS : {[tc["name"] for tc in response.tool_calls]}")

    return {"messages" : list(state.get("messages",[])) + [user_message, response]}

def should_continue(state : State) -> str:
    """Determines whether to continue the conversation or not"""

    for message in reversed(state["messages"]):
        if (isinstance(message, ToolMessage) and 
            "saved" in message.content.lower() and "document" in message.content.lower()):
            return "end"

    return "continue"

graph = StateGraph(State)
graph.add_node("agent", agent)
graph.add_node("tools", ToolNode(tools))

graph.set_entry_point("agent")
graph.set_finish_point("agent")
graph.add_edge("agent", "tools")

graph.add_conditional_edges(
    "tools",
    should_continue,
    {
        "continue" : "agent",
        "end" : END
    }
)
app = graph.compile()

def print_messages(messages):    
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")

def run_agent():
    print("\n =========START========")

    state = {"messages" : []}
    for step in app.stream(state, stream_mode="values"):
        if ("messages" in step):
            print_messages(step["messages"])

    print("\n =========END========")

if (__name__ == "__main__"):
    run_agent()