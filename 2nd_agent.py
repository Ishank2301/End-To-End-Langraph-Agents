import os
from typing import TypedDict, List, Union
from langchain_core.messages import HumanMessage, AIMessage
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
    message: List[Union[HumanMessage, AIMessage]]

llm = ChatOllama(model="llama3")

def process(state:AgentState) -> AgentState:   # type: ignore
    response = llm.invoke(state["message"])
    
    state["message"].append(AIMessage(content=response.content))
    print(f"\nAI:{response.content}")
    
    return state


graph = StateGraph(AgentState)
graph.add_node("process", process)
graph.add_edge(START, "process")
graph.add_edge("process",  END)
agent = graph.compile()
conversation_history = []



user_input = input("Enter: ")
while user_input != "exit":
    conversation_history.append(HumanMessage(content=user_input))
    
    result = agent.invoke({"message": [HumanMessage(content=user_input)]})
    conversation_history = result["message"]
    user_input = input("Enter: ")


with open("logging.txt", "w") as file:
    file.write("Your conversation Log:\n")
    
    for message in conversation_history:
        if isinstance(message, HumanMessage):
            file.write(f"Human: {message.content}\n")
        elif isinstance(message, AIMessage):
            file.write(f"AI: {message.content}\n\n")
    file.write("End Of Conversation")

print("Conversation saved to logging.txt")
            