import os
from typing import TypedDict, List
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.graph import StateGraph, START, END
import os
from dotenv import load_dotenv

# Load .env file        
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

# Just to check (optional, remove later)
llm = ChatOllama(model=ollama_model, base_url=ollama_host)


class AgentState(TypedDict):
    message: List[HumanMessage]

llm = ChatOllama(model="llama3")


def process(state: AgentState) -> AgentState:
    response = llm.invoke(state["message"])
    print(f"\nAI: {response.content}")
    return state

graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process", END)

agent = graph.compile()

user_input = input("Enter: ")
while user_input != "exit":
    agent.invoke({"message": [HumanMessage(content=user_input)]})
    user_input = input("Enter: ")
