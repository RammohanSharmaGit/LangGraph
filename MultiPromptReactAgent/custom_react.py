from typing import TypedDict, Sequence, Annotated
from dotenv import load_dotenv
import os
from langchain_core.messages import BaseMessage, ToolMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langchain_core.tools import tool


class State(TypedDict):
    messages : Annotated[Sequence[BaseMessage], add_messages]


load_dotenv()
key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model = "gpt-5-mini", api_key= key)

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
    
def make_custom_react_agent(tools) :
    global llm
    llm = llm.bind_tools(tools)
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
    return app

