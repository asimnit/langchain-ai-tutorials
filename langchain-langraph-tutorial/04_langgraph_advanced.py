"""
LangGraph Advanced Tutorial
===========================
Build production-ready agents with LangGraph.
Includes: ReAct Agent, Multi-Agent Systems, Persistence, and Human-in-the-loop.
Using Claude on Azure AI Foundry as the LLM provider.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Sequence
from operator import add
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, SystemMessage
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()

# Initialize the model - Claude on Azure AI Foundry
llm = ChatAnthropic(
    model=os.getenv("AZURE_ANTHROPIC_MODEL", "claude-opus-4-6"),
    anthropic_api_url=os.getenv("AZURE_ANTHROPIC_ENDPOINT"),
    api_key=os.getenv("AZURE_ANTHROPIC_API_KEY"),
    max_tokens=1024,
    temperature=0,
)


# =============================================================================
# 1. ReAct AGENT - Reason + Act pattern
# =============================================================================

# Define tools
@tool
def search_web(query: str) -> str:
    """Search the web for information."""
    # Simulated search results
    results = {
        "weather": "Current weather: Sunny, 72°F",
        "news": "Top headline: AI advances continue",
        "stocks": "Market update: S&P 500 up 1.2%"
    }
    for key, value in results.items():
        if key in query.lower():
            return value
    return f"Search results for '{query}': No specific data found."


@tool
def calculate(expression: str) -> str:
    """Perform mathematical calculations."""
    try:
        return str(eval(expression))
    except:
        return "Error in calculation"


@tool
def get_current_time() -> str:
    """Get the current date and time."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


tools = [search_web, calculate, get_current_time]


# State definition for the agent
class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]


# Bind tools to the model
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: AgentState) -> dict:
    """The agent decides what to do next."""
    messages = state["messages"]
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Check if we should use tools or end."""
    last_message = state["messages"][-1]
    
    # If the AI wants to use tools, continue
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# Create tool node (handles tool execution automatically)
tool_node = ToolNode(tools)

# Build the ReAct agent graph
agent_builder = StateGraph(AgentState)

agent_builder.add_node("agent", agent_node)
agent_builder.add_node("tools", tool_node)

agent_builder.add_edge(START, "agent")
agent_builder.add_conditional_edges(
    "agent",
    should_continue,
    {
        "tools": "tools",
        "end": END
    }
)
agent_builder.add_edge("tools", "agent")  # After tools, go back to agent

react_agent = agent_builder.compile()

# Test the ReAct agent
print("--- ReAct Agent ---")
result = react_agent.invoke({
    "messages": [HumanMessage(content="What's 25 * 47? Also, what time is it?")]
})
for msg in result["messages"]:
    print(f"{type(msg).__name__}: {msg.content[:100] if msg.content else msg.tool_calls}")


# =============================================================================
# 2. PERSISTENCE - Save and resume conversations
# =============================================================================

# Create a memory saver (in-memory for demo, use SqliteSaver for production)
memory = MemorySaver()

# Compile with checkpointing
persistent_agent = agent_builder.compile(checkpointer=memory)

# Use thread_id to maintain conversation state
config_1 = {"configurable": {"thread_id": "user_123"}}

print("\n--- Persistent Agent ---")

# First message in conversation
result1 = persistent_agent.invoke(
    {"messages": [HumanMessage(content="Hi, my name is Alice. I'm learning Python.")]},
    config=config_1
)
print("Turn 1:", result1["messages"][-1].content[:100])

# Second message - agent remembers context!
result2 = persistent_agent.invoke(
    {"messages": [HumanMessage(content="What's my name and what am I learning?")]},
    config=config_1
)
print("Turn 2:", result2["messages"][-1].content[:100])

# Different thread = different conversation
config_2 = {"configurable": {"thread_id": "user_456"}}
result3 = persistent_agent.invoke(
    {"messages": [HumanMessage(content="What's my name?")]},
    config=config_2
)
print("Different thread:", result3["messages"][-1].content[:100])


# =============================================================================
# 3. HUMAN-IN-THE-LOOP - Require approval for dangerous actions
# =============================================================================

@tool
def send_email(to: str, subject: str, body: str) -> str:
    """Send an email (requires human approval)."""
    return f"Email sent to {to}"


@tool
def delete_data(table_name: str) -> str:
    """Delete data from database (DANGEROUS - requires approval)."""
    return f"Data deleted from {table_name}"


dangerous_tools = [send_email, delete_data, search_web]
llm_dangerous = llm.bind_tools(dangerous_tools)

# Define which tools need approval
TOOLS_REQUIRING_APPROVAL = {"send_email", "delete_data"}


class HumanApprovalState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    pending_approval: dict | None
    approved: bool


def approval_agent(state: HumanApprovalState) -> dict:
    """Agent node with approval awareness."""
    messages = state["messages"]
    response = llm_dangerous.invoke(messages)
    
    # Check if any tool calls need approval
    pending = None
    if hasattr(response, "tool_calls") and response.tool_calls:
        for tc in response.tool_calls:
            if tc["name"] in TOOLS_REQUIRING_APPROVAL:
                pending = tc
                break
    
    return {
        "messages": [response],
        "pending_approval": pending,
        "approved": False
    }


def check_approval(state: HumanApprovalState) -> Literal["wait_approval", "execute", "end"]:
    """Route based on whether approval is needed."""
    last_message = state["messages"][-1]
    
    if not hasattr(last_message, "tool_calls") or not last_message.tool_calls:
        return "end"
    
    if state.get("pending_approval"):
        return "wait_approval"
    
    return "execute"


def human_approval_node(state: HumanApprovalState) -> dict:
    """Wait for human approval (in production, this would pause and wait)."""
    pending = state["pending_approval"]
    
    print(f"\n⚠️  APPROVAL REQUIRED for: {pending['name']}")
    print(f"   Arguments: {pending['args']}")
    
    # Simulate getting approval (in production, this would be async)
    approval = input("   Approve? (yes/no): ")
    
    return {"approved": approval.lower() == "yes"}


def execute_or_cancel(state: HumanApprovalState) -> dict:
    """Execute the tool or cancel based on approval."""
    if not state.get("approved"):
        return {
            "messages": [AIMessage(content="Operation cancelled by user.")],
            "pending_approval": None
        }
    
    # Execute the approved tool
    pending = state["pending_approval"]
    tool_map = {t.name: t for t in dangerous_tools}
    result = tool_map[pending["name"]].invoke(pending["args"])
    
    return {
        "messages": [ToolMessage(content=result, tool_call_id=pending["id"])],
        "pending_approval": None
    }


# Build approval workflow
approval_builder = StateGraph(HumanApprovalState)
approval_builder.add_node("agent", approval_agent)
approval_builder.add_node("get_approval", human_approval_node)
approval_builder.add_node("execute", execute_or_cancel)
approval_builder.add_node("tools", ToolNode(dangerous_tools))

approval_builder.add_edge(START, "agent")
approval_builder.add_conditional_edges(
    "agent",
    check_approval,
    {
        "wait_approval": "get_approval",
        "execute": "tools",
        "end": END
    }
)
approval_builder.add_edge("get_approval", "execute")
approval_builder.add_edge("execute", "agent")
approval_builder.add_edge("tools", "agent")

# approval_agent_graph = approval_builder.compile()
# Uncomment and test:
# result = approval_agent_graph.invoke({
#     "messages": [HumanMessage(content="Send an email to bob@example.com saying hello")],
#     "pending_approval": None,
#     "approved": False
# })


# =============================================================================
# 4. MULTI-AGENT SYSTEM - Supervisor Pattern
# =============================================================================

class MultiAgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    next_agent: str


def researcher_agent(state: MultiAgentState) -> dict:
    """Agent specialized in research."""
    prompt = """You are a research specialist. Analyze the request and provide 
    relevant information. Be concise and factual."""
    
    messages = [SystemMessage(content=prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    
    return {"messages": [AIMessage(content=f"[RESEARCHER]: {response.content}")]}


def writer_agent(state: MultiAgentState) -> dict:
    """Agent specialized in writing."""
    prompt = """You are a writing specialist. Take the research provided and 
    create well-structured, engaging content."""
    
    messages = [SystemMessage(content=prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    
    return {"messages": [AIMessage(content=f"[WRITER]: {response.content}")]}


def reviewer_agent(state: MultiAgentState) -> dict:
    """Agent that reviews and provides feedback."""
    prompt = """You are a quality reviewer. Review the content and provide 
    constructive feedback or approve it."""
    
    messages = [SystemMessage(content=prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    
    return {"messages": [AIMessage(content=f"[REVIEWER]: {response.content}")]}


def supervisor(state: MultiAgentState) -> dict:
    """Supervisor decides which agent works next."""
    prompt = """You are a team supervisor. Based on the conversation, decide 
    which team member should work next:
    - 'researcher': For gathering information
    - 'writer': For creating content
    - 'reviewer': For reviewing work
    - 'FINISH': When the task is complete
    
    Respond with just the agent name."""
    
    messages = [SystemMessage(content=prompt)] + list(state["messages"])
    response = llm.invoke(messages)
    
    next_agent = response.content.strip().lower()
    
    # Validate response
    valid_agents = ["researcher", "writer", "reviewer", "finish"]
    if next_agent not in valid_agents:
        next_agent = "researcher"  # Default
    
    print(f"→ Supervisor assigns: {next_agent}")
    return {"next_agent": next_agent}


def route_to_agent(state: MultiAgentState) -> str:
    """Route to the appropriate agent."""
    next_agent = state.get("next_agent", "researcher")
    if next_agent == "finish":
        return "end"
    return next_agent


# Build multi-agent graph
multi_agent_builder = StateGraph(MultiAgentState)

multi_agent_builder.add_node("supervisor", supervisor)
multi_agent_builder.add_node("researcher", researcher_agent)
multi_agent_builder.add_node("writer", writer_agent)
multi_agent_builder.add_node("reviewer", reviewer_agent)

multi_agent_builder.add_edge(START, "supervisor")
multi_agent_builder.add_conditional_edges(
    "supervisor",
    route_to_agent,
    {
        "researcher": "researcher",
        "writer": "writer",
        "reviewer": "reviewer",
        "end": END
    }
)

# All agents report back to supervisor
multi_agent_builder.add_edge("researcher", "supervisor")
multi_agent_builder.add_edge("writer", "supervisor")
multi_agent_builder.add_edge("reviewer", "supervisor")

multi_agent_graph = multi_agent_builder.compile()

# Test multi-agent system
print("\n--- Multi-Agent System ---")
result = multi_agent_graph.invoke({
    "messages": [HumanMessage(content="Create a short blog post about the benefits of meditation.")],
    "next_agent": ""
})

print("\nFinal Output:")
for msg in result["messages"][-3:]:
    print(msg.content[:200], "...\n")


# =============================================================================
# 5. SUBGRAPHS - Compose graphs together
# =============================================================================

class SubGraphState(TypedDict):
    input: str
    processed: str
    output: str


# Create a subgraph for text processing
def preprocess(state: SubGraphState) -> dict:
    """Clean and prepare text."""
    cleaned = state["input"].strip().lower()
    return {"processed": cleaned}


def analyze(state: SubGraphState) -> dict:
    """Analyze the text."""
    text = state["processed"]
    analysis = f"Analyzed: {len(text)} chars, {len(text.split())} words"
    return {"output": analysis}


# Build subgraph
sub_builder = StateGraph(SubGraphState)
sub_builder.add_node("preprocess", preprocess)
sub_builder.add_node("analyze", analyze)
sub_builder.add_edge(START, "preprocess")
sub_builder.add_edge("preprocess", "analyze")
sub_builder.add_edge("analyze", END)

text_processor = sub_builder.compile()


# Main graph that uses the subgraph
class MainState(TypedDict):
    raw_input: str
    final_result: str


def main_process(state: MainState) -> dict:
    """Use the subgraph for processing."""
    # Call the subgraph
    sub_result = text_processor.invoke({
        "input": state["raw_input"],
        "processed": "",
        "output": ""
    })
    return {"final_result": sub_result["output"]}


main_builder = StateGraph(MainState)
main_builder.add_node("process", main_process)
main_builder.add_edge(START, "process")
main_builder.add_edge("process", END)

main_graph = main_builder.compile()

print("\n--- Subgraph Composition ---")
result = main_graph.invoke({"raw_input": "  Hello World!  ", "final_result": ""})
print("Result:", result["final_result"])
