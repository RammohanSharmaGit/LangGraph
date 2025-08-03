from typing import TypedDict, List
from langgraph.graph import StateGraph, START, END
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import HumanMessage
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    google_api_key = key
)

class State(TypedDict):
    messages : List[HumanMessage]

def process(state: State) -> State:
    response = llm.invoke(state["messages"])
    print(f"AI : {response}")
    return state

graph = StateGraph(State)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

user_input = input("enter your prompt: ")
while (not user_input== "exit"):
    agent.invoke({"messages" : [HumanMessage(content = user_input)]})
    user_input = input("enter your prompt: ")