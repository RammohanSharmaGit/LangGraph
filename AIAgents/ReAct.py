from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

llm = ChatGoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    google_api_key = key
)

class State(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]

@tool
def add(a : int, b : int):
    """function to add 2 numbers"""
    return a + b

@tool
def subtract(a : int, b : int):
    """function to subtract 2 numbers"""
    return a - b

@tool
def multiply(a : int, b : int):
    """function to multiply 2 numbers"""
    return a * b

tools = [add, subtract, multiply]

model = llm.bind_tools(tools)

def agent_call(state : State) -> State:
    system_prompt = SystemMessage(
        content = "You are my AI assistant, please answer my query to the best of your ability."
    )
    response = llm.invoke([system_prompt] + state["messages"])
    return {"messages":response}

def should_continue(state : State):
    if not state["messages"][-1].tool_calls:
        return "end"
    else:
        return "continue"
    
graph = StateGraph(State)
graph.add_node("agent", agent_call)

tool_node = ToolNode(tools = tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("agent")
graph.add_conditional_edges(
    "agent",
    should_continue,
    {
        "end" : END,
        "continue" : "tools"
    }
)
graph.add_edge("tools", "agent")

app = graph.compile()

#function to print
def print_stream(stream):
    for s in stream :
        message = s["messages"][-1]
        if isinstance(message, tuple):
            print(message)
        else:
            message.pretty_print()

input = {"messages" : "Add 4 and 3, multiply the result by 5 and subtract 5 from it"}
print_stream(app.stream(input, stream_mode= "values"))

