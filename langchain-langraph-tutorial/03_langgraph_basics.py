"""
LangGraph Basics Tutorial
=========================
LangGraph uses a graph structure: Nodes (functions) + Edges (connections)
This enables complex workflows with loops, branching, and state management.
Using Claude on Azure AI Foundry as the LLM provider.
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal
from langchain_anthropic import ChatAnthropic
from langgraph.graph import StateGraph, END, START

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
# 1. STATE - The data that flows through your graph
# =============================================================================

# Define what data your graph will track
class SimpleState(TypedDict):
    """State that gets passed between nodes."""
    messages: list[str]      # Chat messages
    current_step: str        # Track where we are
    result: str              # Final output


# =============================================================================
# 2. NODES - Functions that process the state
# =============================================================================

def step_one(state: SimpleState) -> SimpleState:
    """First node: Initialize and process."""
    print("→ Executing Step One")
    
    # Read from state
    messages = state.get("messages", [])
    
    # Modify and return updates
    return {
        "messages": messages + ["Step one completed"],
        "current_step": "step_one_done"
    }


def step_two(state: SimpleState) -> SimpleState:
    """Second node: Continue processing."""
    print("→ Executing Step Two")
    
    messages = state.get("messages", [])
    
    return {
        "messages": messages + ["Step two completed"],
        "current_step": "step_two_done",
        "result": "All steps completed!"
    }


# =============================================================================
# 3. BUILDING THE GRAPH - Connect nodes with edges
# =============================================================================

# Create a graph builder
builder = StateGraph(SimpleState)

# Add nodes
builder.add_node("step_one", step_one)
builder.add_node("step_two", step_two)

# Add edges (connections between nodes)
builder.add_edge(START, "step_one")      # Start → step_one
builder.add_edge("step_one", "step_two") # step_one → step_two
builder.add_edge("step_two", END)        # step_two → End

# Compile the graph
simple_graph = builder.compile()

# Run it!
print("--- Simple Graph ---")
initial_state = {"messages": ["Hello"], "current_step": "start", "result": ""}
final_state = simple_graph.invoke(initial_state)
print("Final State:", final_state)


# =============================================================================
# 4. CONDITIONAL EDGES - Branching based on state
# =============================================================================

class RouterState(TypedDict):
    query: str
    query_type: str     # "technical" or "general"
    response: str


def classify_query(state: RouterState) -> RouterState:
    """Classify the incoming query."""
    query = state["query"].lower()
    
    # Simple classification logic
    technical_keywords = ["code", "programming", "bug", "error", "function", "api"]
    
    if any(word in query for word in technical_keywords):
        query_type = "technical"
    else:
        query_type = "general"
    
    print(f"→ Classified as: {query_type}")
    return {"query_type": query_type}


def handle_technical(state: RouterState) -> RouterState:
    """Handle technical queries."""
    print("→ Technical Handler")
    response = llm.invoke(f"As a technical expert, answer: {state['query']}")
    return {"response": response.content}


def handle_general(state: RouterState) -> RouterState:
    """Handle general queries."""
    print("→ General Handler")
    response = llm.invoke(f"Answer this friendly question: {state['query']}")
    return {"response": response.content}


# Routing function - determines which edge to take
def route_query(state: RouterState) -> Literal["handle_technical", "handle_general"]:
    """Decide which handler to use."""
    if state["query_type"] == "technical":
        return "handle_technical"
    return "handle_general"


# Build the conditional graph
router_builder = StateGraph(RouterState)

router_builder.add_node("classify", classify_query)
router_builder.add_node("handle_technical", handle_technical)
router_builder.add_node("handle_general", handle_general)

router_builder.add_edge(START, "classify")

# Conditional edge - branches based on route_query function
router_builder.add_conditional_edges(
    "classify",                    # From node
    route_query,                   # Function that returns the next node name
    {                              # Map of possible destinations
        "handle_technical": "handle_technical",
        "handle_general": "handle_general"
    }
)

router_builder.add_edge("handle_technical", END)
router_builder.add_edge("handle_general", END)

router_graph = router_builder.compile()

# Test the router
print("\n--- Conditional Routing ---")
result1 = router_graph.invoke({"query": "How do I fix a Python error?", "query_type": "", "response": ""})
print("Technical Query Result:", result1["response"][:100], "...")

result2 = router_graph.invoke({"query": "What's a good movie to watch?", "query_type": "", "response": ""})
print("\nGeneral Query Result:", result2["response"][:100], "...")


# =============================================================================
# 5. LOOPS - Graphs that can cycle
# =============================================================================

from operator import add

class LoopState(TypedDict):
    counter: int
    max_count: int
    log: Annotated[list[str], add]  # Lists with 'add' annotation are appended


def increment(state: LoopState) -> LoopState:
    """Increment the counter."""
    new_count = state["counter"] + 1
    print(f"→ Counter: {new_count}")
    return {
        "counter": new_count,
        "log": [f"Incremented to {new_count}"]
    }


def should_continue(state: LoopState) -> Literal["increment", "end"]:
    """Check if we should keep looping."""
    if state["counter"] < state["max_count"]:
        return "increment"
    return "end"


def finish(state: LoopState) -> LoopState:
    """Final node."""
    return {"log": ["Loop completed!"]}


# Build loop graph
loop_builder = StateGraph(LoopState)

loop_builder.add_node("increment", increment)
loop_builder.add_node("finish", finish)

loop_builder.add_edge(START, "increment")

# This creates a cycle!
loop_builder.add_conditional_edges(
    "increment",
    should_continue,
    {
        "increment": "increment",  # Loop back
        "end": "finish"            # Exit loop
    }
)

loop_builder.add_edge("finish", END)

loop_graph = loop_builder.compile()

print("\n--- Loop Graph ---")
result = loop_graph.invoke({
    "counter": 0,
    "max_count": 5,
    "log": ["Starting loop"]
})
print("Final counter:", result["counter"])
print("Log:", result["log"])


# =============================================================================
# 6. STREAMING - Watch execution in real-time
# =============================================================================

print("\n--- Streaming Execution ---")
for event in router_graph.stream(
    {"query": "Explain recursion in programming", "query_type": "", "response": ""},
    stream_mode="updates"  # Shows state updates at each step
):
    print(f"Event: {event}")


# =============================================================================
# 7. VISUALIZATION - See your graph structure
# =============================================================================

# Get the graph as a Mermaid diagram (can render in GitHub, etc.)
print("\n--- Graph Visualization (Mermaid) ---")
try:
    mermaid_png = router_graph.get_graph().draw_mermaid()
    print(mermaid_png)
except Exception as e:
    print(f"Install graphviz for visualization: {e}")
