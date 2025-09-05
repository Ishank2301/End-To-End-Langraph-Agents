import json
from typing import Annotated, Sequence, TypedDict
from dotenv import load_dotenv  
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_ollama import ChatOllama
from langchain_core.tools import tool
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, END
from langchain_ollama import OllamaEmbeddings
from langchain_community.document_loaders import PyPdfLoader # type: ignore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma # type: ignore
from langchain_core.tools import tools
load_dotenv()

# Global variable to store document content
document_content = ""

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]


# ---------------- TOOLS ----------------
@tool
def update(content: str) -> str:
    """Updates the document with the provided content."""
    global document_content
    document_content = content
    return f"✅ Document updated successfully!\nCurrent content:\n{document_content}"


@tool
def save(filename: str) -> str:
    """Saves the current document to a text file and finish the process."""
    global document_content
    if not filename.endswith('.txt'):
        filename = f"{filename}.txt"
    try:
        with open(filename, 'w') as file:
            file.write(document_content)
        print(f"\n💾 Document saved to: {filename}")
        return f"Document has been saved successfully to '{filename}'."
    except Exception as e:
        return f"❌ Error saving document: {str(e)}"


tools = {"update": update, "save": save}

# ✅ Ollama model (no bind_tools)
model = ChatOllama(model="llama3", base_url="http://localhost:11434")


# ---------------- AGENT ----------------
def our_agent(state: AgentState) -> AgentState:
    system_prompt = SystemMessage(content=f"""
    You are Drafter, a helpful writing assistant. 
    You can update or save documents.

    When you want to use a tool, reply ONLY in JSON like this:
    - To update: {{"tool": "update", "args": {{"content": "new content here"}}}}
    - To save:   {{"tool": "save", "args": {{"filename": "mydoc.txt"}}}}

    Otherwise, respond with plain helpful text.
    
    Current document content: {document_content}
    """)

    if not state["messages"]:
        user_input = "I'm ready to help you update a document. What would you like to create?"
        user_message = HumanMessage(content=user_input)
    else:
        user_input = input("\nWhat would you like to do with the document? ")
        print(f"\n👤 USER: {user_input}")
        user_message = HumanMessage(content=user_input)

    all_messages = [system_prompt] + list(state["messages"]) + [user_message]

    response = model.invoke(all_messages)
    print(f"\n🤖 AI: {response.content}")

    # Try parsing tool call
    try:
        data = json.loads(response.content)
        if "tool" in data and data["tool"] in tools:
            tool_name = data["tool"]
            args = data.get("args", {})
            tool_result = tools[tool_name](**args)
            tool_msg = ToolMessage(content=tool_result, name=tool_name)
            return {"messages": list(state["messages"]) + [user_message, response, tool_msg]}
    except Exception:
        pass

    return {"messages": list(state["messages"]) + [user_message, response]}


# ---------------- CONTROL FLOW ----------------
def should_continue(state: AgentState) -> str:
    messages = state["messages"]
    if not messages:
        return "continue"
    for message in reversed(messages):
        if (isinstance(message, ToolMessage) and
            "saved" in message.content.lower() and
            "document" in message.content.lower()):
            return "end"
    return "continue"


def print_messages(messages):
    if not messages:
        return
    for message in messages[-3:]:
        if isinstance(message, ToolMessage):
            print(f"\n🛠️ TOOL RESULT: {message.content}")


# ---------------- GRAPH ----------------
graph = StateGraph(AgentState)
graph.add_node("agent", our_agent)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"continue": "agent", "end": END})
app = graph.compile()


# ---------------- RUN ----------------
def run_document_agent():
    print("\n===== DRAFTER =====")
    state = {"messages": []}
    for step in app.stream(state, stream_mode="values"):
        if "messages" in step:
            print_messages(step["messages"])
    print("\n===== DRAFTER FINISHED =====")


if __name__ == "__main__":
    run_document_agent()
