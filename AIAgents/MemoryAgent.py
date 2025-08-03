from typing import TypedDict, List, Union
import os
from dotenv import load_dotenv
from langgraph.graph import StateGraph, END, START
from langchain_google_genai import GoogleGenerativeAI
from langchain_core.messages import AIMessage, HumanMessage

load_dotenv()
key = os.getenv("GEMINI_API_KEY")

llm = GoogleGenerativeAI(
    model = "gemini-2.5-flash-lite",
    google_api_key = key
)

class State(TypedDict):
    messages: List[Union[HumanMessage, AIMessage]]

def process(state : State) -> State:
    response = llm.invoke(state["messages"])
    state["messages"].append(AIMessage(content=response))
    print(response)
    print(f"current state : {state["messages"]}")

graph = StateGraph(State)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)
agent = graph.compile()

conversation_history : List[Union[AIMessage, HumanMessage]] = []

user_input = input("Enter your prompt: ")
while (not user_input == "exit"):
    conversation_history.append(HumanMessage(user_input))
    output = agent.invoke({"messages" : conversation_history})
    conversation_history = output["messages"]
    user_input = input("Enter your prompt: ")
