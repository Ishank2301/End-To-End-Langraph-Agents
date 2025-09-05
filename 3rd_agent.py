import os
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  

from langchain_core.messages import BaseMessage, AIMessage, HumanMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode


# Load environment variables
load_dotenv()

ollama_host = os.getenv("OLLAMA_HOST", "http://localhost:11434")
ollama_model = os.getenv("OLLAMA_MODEL", "llama3")

# Initialize Ollama model
llm = ChatOllama(model=ollama_model, base_url=ollama_host)

# State definition for LangGraph
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# Define tools
@tool
def add(a: int, b: int):
    """Add two numbers together"""
    return a + b

@tool
def subtract(a: int, b: int):
    """Subtract b from a"""
    return a - b

@tool
def multiply(a: int, b: int):
    """Multiply two numbers"""
    return a * b


tools = [add, subtract, multiply]

# Bind tools to the model
model = llm 


# Model call node
def model_call(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(
        content="You are my AI assistant, please answer my query to the best of your ability."
    )
    response = model.invoke([system_prompt] + state["messages"])  # type: ignore
    return {"messages": [response]}


# Decide whether to continue or end
def should_continue(state: AgentState):
    last_message = state["messages"][-1]
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "continue"
    return "end"


# Build the LangGraph
graph = StateGraph(AgentState)
graph.add_node("our_agent", model_call)

tool_node = ToolNode(tools=tools)
graph.add_node("tools", tool_node)

graph.set_entry_point("our_agent")

graph.add_conditional_edges(
    "our_agent",
    should_continue,
    {
        "continue": "tools",
        "end": END,
    },
)

graph.add_edge("tools", "our_agent")

app = graph.compile()


# Stream printing utility
def print_stream(stream):
    for s in stream:
        message = s["messages"][-1]
        if isinstance(message, BaseMessage):
            message.pretty_print()


# Input from user
inputs = {
    "messages": [
        HumanMessage(content="Add 40 + 12 and then multiply the result by 6. Also tell me a joke please.")
    ]
}

# Run the agent
print_stream(app.stream(inputs, stream_mode="values"))  # type: ignore
