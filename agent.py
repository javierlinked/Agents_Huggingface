from dotenv import load_dotenv
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import tools_condition
from langgraph.prebuilt import ToolNode
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

load_dotenv()

from tools import get_tools

with open("system_prompt.txt", "r", encoding="utf-8") as f:
    system_prompt = f.read()

sys_msg = SystemMessage(content=system_prompt)

def build_graph(provider: str):
    # Initialize LLM based on provider
    if provider == "google":
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
    elif provider == "groq":
        llm = ChatGroq(model="qwen-qwq-32b", temperature=0.1,verbose=True)
    else:
        raise ValueError("Invalid provider. Choose 'google' or 'groq'.")
    
    tools = get_tools()
    llm_with_tools = llm.bind_tools(tools)

    def assistant(state: MessagesState):
        """Assistant node - handles reasoning and tool calling"""
        try:
            response = llm_with_tools.invoke(state["messages"])
            return {"messages": [response]}
        except Exception as e:
            error_msg = HumanMessage(content=f"Error in assistant: {str(e)}")
            return {"messages": [error_msg]}

    def retriever(state: MessagesState):
        """Retriever node - adds system message"""
        messages = [sys_msg] + [msg for msg in state["messages"] if not isinstance(msg, SystemMessage)]
        return {"messages": messages}

    # Define the state graph
    graph = StateGraph(MessagesState)
    graph.add_node("retriever", retriever)
    graph.add_node("assistant", assistant)
    graph.add_node("tools", ToolNode(tools))
    
    # Graph flow
    graph.add_edge(START, "retriever")
    graph.add_edge("retriever", "assistant")
    graph.add_conditional_edges("assistant", tools_condition)
    graph.add_edge("tools", "assistant")
    
    return graph.compile()
    