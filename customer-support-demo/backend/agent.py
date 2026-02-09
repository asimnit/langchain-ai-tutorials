"""
Customer Support Agent with LangGraph
Streaming support for WebSocket communication
"""

import os
import json
from typing import TypedDict, Annotated, Literal, Sequence, AsyncGenerator
from operator import add
from dotenv import load_dotenv

from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# =============================================================================
# SETUP: Models and Embeddings
# =============================================================================

llm = ChatAnthropic(
    model=os.getenv("AZURE_ANTHROPIC_MODEL", "claude-opus-4-6"),
    anthropic_api_url=os.getenv("AZURE_ANTHROPIC_ENDPOINT"),
    api_key=os.getenv("AZURE_ANTHROPIC_API_KEY"),
    max_tokens=1024,
    temperature=0.3,
)

embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)


# =============================================================================
# KNOWLEDGE BASE
# =============================================================================

knowledge_base = [
    Document(
        page_content="Return Policy: Items can be returned within 30 days of purchase. Items must be unused and in original packaging. Refunds are processed within 5-7 business days.",
        metadata={"topic": "returns"}
    ),
    Document(
        page_content="Shipping: Standard shipping takes 5-7 business days. Express shipping takes 2-3 business days. Free shipping on orders over $50.",
        metadata={"topic": "shipping"}
    ),
    Document(
        page_content="Payment Methods: We accept Visa, MasterCard, American Express, PayPal, and Apple Pay. All transactions are secured with SSL encryption.",
        metadata={"topic": "payment"}
    ),
    Document(
        page_content="Account Issues: To reset your password, click 'Forgot Password' on the login page. For locked accounts, contact support with your registered email.",
        metadata={"topic": "account"}
    ),
    Document(
        page_content="Product Warranty: All electronics come with a 1-year manufacturer warranty. Extended warranties are available for purchase.",
        metadata={"topic": "warranty"}
    ),
]

vectorstore = InMemoryVectorStore.from_documents(knowledge_base, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


# =============================================================================
# SIMULATED ORDER DATABASE
# =============================================================================

ORDERS_DB = {
    "ORD-001": {"status": "shipped", "tracking": "1Z999AA10123456784", "items": ["Laptop"], "total": 999.99},
    "ORD-002": {"status": "processing", "tracking": None, "items": ["Mouse", "Keyboard"], "total": 89.99},
    "ORD-003": {"status": "delivered", "tracking": "1Z999AA10987654321", "items": ["Monitor"], "total": 299.99},
}


# =============================================================================
# TOOLS
# =============================================================================

@tool
def lookup_order(order_id: str) -> str:
    """Look up an order by its ID to get status, tracking, and details."""
    order = ORDERS_DB.get(order_id.upper())
    if order:
        return json.dumps({
            "order_id": order_id.upper(),
            "status": order["status"],
            "tracking": order["tracking"] or "Not yet available",
            "items": order["items"],
            "total": f"${order['total']}"
        }, indent=2)
    return f"Order {order_id} not found. Please check the order ID."


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for information about policies, shipping, returns, etc."""
    docs = retriever.invoke(query)
    if docs:
        return "\n\n".join([f"• {doc.page_content}" for doc in docs])
    return "No relevant information found in the knowledge base."


@tool
def check_return_eligibility(order_id: str) -> str:
    """Check if an order is eligible for return based on the order status."""
    order = ORDERS_DB.get(order_id.upper())
    if not order:
        return f"Order {order_id} not found."
    
    if order["status"] == "delivered":
        return json.dumps({
            "order_id": order_id.upper(),
            "eligible": True,
            "reason": "Order has been delivered and is within the 30-day return window.",
            "next_steps": "To initiate a return, please visit our returns portal or contact support."
        }, indent=2)
    elif order["status"] == "shipped":
        return json.dumps({
            "order_id": order_id.upper(),
            "eligible": False,
            "reason": "Order is still in transit. Please wait until delivery to initiate a return.",
            "tracking": order["tracking"]
        }, indent=2)
    else:
        return json.dumps({
            "order_id": order_id.upper(),
            "eligible": False,
            "reason": "Order is still being processed. It can be cancelled instead of returned.",
            "next_steps": "Would you like to cancel this order?"
        }, indent=2)


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human agent when the issue is complex or the customer requests it."""
    return json.dumps({
        "escalated": True,
        "reason": reason,
        "message": "I've escalated your case to a human agent. They will contact you within 24 hours via email.",
        "ticket_id": "TKT-" + str(hash(reason))[-6:]
    }, indent=2)


tools = [lookup_order, search_knowledge_base, check_return_eligibility, escalate_to_human]
llm_with_tools = llm.bind_tools(tools)


# =============================================================================
# AGENT STATE
# =============================================================================

class AgentState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]


# =============================================================================
# AGENT NODES
# =============================================================================

SYSTEM_PROMPT = """You are a helpful customer support agent for an e-commerce company. 
You help customers with:
- Order status and tracking
- Return and refund requests
- Shipping information
- Payment questions
- Account issues

Always be polite, professional, and helpful. Use the available tools to look up information.
If you cannot resolve an issue, offer to escalate to a human agent.

When searching the knowledge base, provide clear and concise answers based on the information found.
When looking up orders, provide all relevant details to the customer."""


def call_model(state: AgentState) -> AgentState:
    """Call the LLM with the current messages."""
    messages = state["messages"]
    
    # Add system prompt if not present
    if not any(isinstance(m, SystemMessage) for m in messages):
        messages = [SystemMessage(content=SYSTEM_PROMPT)] + list(messages)
    
    response = llm_with_tools.invoke(messages)
    return {"messages": [response]}


def should_continue(state: AgentState) -> Literal["tools", "end"]:
    """Determine if we should continue to tools or end."""
    last_message = state["messages"][-1]
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    return "end"


# =============================================================================
# BUILD THE GRAPH
# =============================================================================

# Initialize memory checkpointer for persistent conversation history
memory = MemorySaver()


def create_agent():
    """Create and compile the customer support agent graph with memory."""
    builder = StateGraph(AgentState)
    
    # Add nodes
    builder.add_node("agent", call_model)
    builder.add_node("tools", ToolNode(tools))
    
    # Add edges
    builder.add_edge(START, "agent")
    builder.add_conditional_edges(
        "agent",
        should_continue,
        {"tools": "tools", "end": END}
    )
    builder.add_edge("tools", "agent")
    
    # Compile with memory checkpointer for automatic history management
    return builder.compile(checkpointer=memory)


agent = create_agent()


# =============================================================================
# STREAMING INTERFACE FOR WEBSOCKET
# =============================================================================

async def stream_agent_response(
    user_message: str,
    thread_id: str
) -> AsyncGenerator[dict, None]:
    """
    Stream agent responses as structured events for the frontend.
    Uses LangGraph's MemorySaver for automatic conversation history management.
    
    Event types:
    - thinking: Agent is processing
    - tool_call: Agent is calling a tool
    - tool_result: Tool execution result
    - response: Text response (streaming)
    - done: Conversation turn complete
    - error: An error occurred
    
    Args:
        user_message: The user's message
        thread_id: Unique thread identifier for conversation memory
    """
    
    # Configuration for memory checkpointer
    config = {"configurable": {"thread_id": thread_id}}
    
    # Input with just the new user message - history is managed by checkpointer
    input_messages = [HumanMessage(content=user_message)]
    
    # Check if this is a new conversation and add system prompt
    state = agent.get_state(config)
    if not state.values.get("messages"):
        input_messages = [SystemMessage(content=SYSTEM_PROMPT)] + input_messages
    
    try:
        # Signal thinking
        yield {"type": "thinking", "content": "Analyzing your request..."}
        
        # Stream using astream_events for fine-grained control
        accumulated_content = ""
        
        async for event in agent.astream_events(
            {"messages": input_messages},
            config=config,
            version="v2"
        ):
            kind = event["event"]
            
            # Handle LLM streaming tokens
            if kind == "on_chat_model_stream":
                chunk = event["data"]["chunk"]
                if chunk.content:
                    if isinstance(chunk.content, str):
                        accumulated_content += chunk.content
                        yield {"type": "response", "content": chunk.content, "done": False}
                    elif isinstance(chunk.content, list):
                        for block in chunk.content:
                            if isinstance(block, dict) and block.get("type") == "text":
                                text = block.get("text", "")
                                if text:
                                    accumulated_content += text
                                    yield {"type": "response", "content": text, "done": False}
            
            # Handle tool calls
            elif kind == "on_tool_start":
                tool_input = event["data"].get("input", {})
                # Ensure args are JSON serializable
                if not isinstance(tool_input, dict):
                    tool_input = {"input": str(tool_input)}
                yield {
                    "type": "tool_call",
                    "name": event["name"],
                    "args": tool_input
                }
            
            # Handle tool results
            elif kind == "on_tool_end":
                output = event["data"].get("output", "")
                # Convert to string if it's a message object
                if hasattr(output, "content"):
                    output = output.content
                yield {
                    "type": "tool_result",
                    "name": event["name"],
                    "result": str(output)
                }
                yield {"type": "thinking", "content": "Processing tool results..."}
        
        # Mark streaming as done
        yield {"type": "response", "content": "", "done": True}
        
        # Signal completion
        yield {"type": "done", "messages": [{"role": "assistant", "content": accumulated_content}]}
        
    except Exception as e:
        import traceback
        print(f"[Agent] Error: {e}")
        traceback.print_exc()
        yield {"type": "error", "content": str(e)}


def clear_conversation(thread_id: str) -> None:
    """Clear conversation history for a thread."""
    # Delete all checkpoints for this thread
    config = {"configurable": {"thread_id": thread_id}}
    # Get current state and check if it exists
    state = agent.get_state(config)
    if state.values:
        # Update with empty messages to clear history
        agent.update_state(config, {"messages": []})
