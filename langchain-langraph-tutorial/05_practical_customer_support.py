"""
Practical Example: Customer Support Chatbot
============================================
A complete example combining LangChain + LangGraph to build 
a production-ready customer support system.
Using Claude on Azure AI Foundry as the LLM provider.

Features:
- Intent classification (routes to appropriate handler)
- RAG for knowledge base lookup
- Tool usage for order management
- Conversation memory
- Escalation to human
"""

import os
from dotenv import load_dotenv
from typing import TypedDict, Annotated, Literal, Sequence
from operator import add
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureOpenAIEmbeddings
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, SystemMessage
from langchain_core.tools import tool
from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

load_dotenv()


# =============================================================================
# SETUP: Models, Tools, and Knowledge Base
# =============================================================================

# Initialize the model - Claude on Azure AI Foundry
llm = ChatAnthropic(
    model=os.getenv("AZURE_ANTHROPIC_MODEL", "claude-opus-4-6"),
    anthropic_api_url=os.getenv("AZURE_ANTHROPIC_ENDPOINT"),
    api_key=os.getenv("AZURE_ANTHROPIC_API_KEY"),
    max_tokens=1024,
    temperature=0.3,
)

# Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)

# --- Simulated Knowledge Base ---
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


# --- Simulated Order Database ---
ORDERS_DB = {
    "ORD-001": {"status": "shipped", "tracking": "1Z999AA10123456784", "items": ["Laptop"], "total": 999.99},
    "ORD-002": {"status": "processing", "tracking": None, "items": ["Mouse", "Keyboard"], "total": 89.99},
    "ORD-003": {"status": "delivered", "tracking": "1Z999AA10987654321", "items": ["Monitor"], "total": 299.99},
}


# =============================================================================
# TOOLS: What the agent can do
# =============================================================================

@tool
def lookup_order(order_id: str) -> str:
    """Look up order details by order ID. Use when customer asks about their order."""
    order_id = order_id.upper()
    if order_id in ORDERS_DB:
        order = ORDERS_DB[order_id]
        return f"""Order {order_id}:
        - Status: {order['status']}
        - Items: {', '.join(order['items'])}
        - Total: ${order['total']}
        - Tracking: {order['tracking'] or 'Not yet available'}"""
    return f"Order {order_id} not found. Please check the order ID."


@tool  
def initiate_return(order_id: str, reason: str) -> str:
    """Start a return process for an order. Use when customer wants to return items."""
    order_id = order_id.upper()
    if order_id not in ORDERS_DB:
        return f"Order {order_id} not found."
    
    order = ORDERS_DB[order_id]
    if order["status"] != "delivered":
        return f"Cannot initiate return. Order status is '{order['status']}'. Returns only available for delivered orders."
    
    return f"Return initiated for {order_id}. Return label will be emailed within 24 hours. Reason recorded: {reason}"


@tool
def search_knowledge_base(query: str) -> str:
    """Search company knowledge base for policies, FAQs, and information."""
    docs = retriever.invoke(query)
    if docs:
        return "\n\n".join([doc.page_content for doc in docs])
    return "No relevant information found in knowledge base."


@tool
def escalate_to_human(reason: str) -> str:
    """Escalate the conversation to a human agent. Use for complex issues or angry customers."""
    return f"[ESCALATED] Conversation transferred to human agent. Reason: {reason}. A support representative will respond within 10 minutes."


tools = [lookup_order, initiate_return, search_knowledge_base, escalate_to_human]


# =============================================================================
# STATE: Conversation tracking
# =============================================================================

class SupportState(TypedDict):
    messages: Annotated[Sequence[BaseMessage], add]
    intent: str
    sentiment: str
    escalated: bool


# =============================================================================
# NODES: Processing steps
# =============================================================================

def classify_intent(state: SupportState) -> dict:
    """Classify the customer's intent."""
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    classification_prompt = ChatPromptTemplate.from_template("""
    Classify this customer message into one of these intents:
    - order_inquiry: Questions about orders, tracking, status
    - return_request: Wants to return or exchange items
    - policy_question: Questions about policies, shipping, payment, etc.
    - complaint: Unhappy, frustrated, or complaining
    - general: General questions or greetings
    
    Message: {message}
    
    Respond with just the intent name.""")
    
    chain = classification_prompt | llm
    result = chain.invoke({"message": last_message})
    intent = result.content.strip().lower()
    
    print(f"→ Intent classified: {intent}")
    return {"intent": intent}


def analyze_sentiment(state: SupportState) -> dict:
    """Analyze customer sentiment."""
    last_message = state["messages"][-1].content if state["messages"] else ""
    
    sentiment_prompt = ChatPromptTemplate.from_template("""
    Analyze the sentiment of this customer message:
    - positive: Happy, satisfied, polite
    - neutral: Normal inquiry, no strong emotion
    - negative: Frustrated, angry, upset
    
    Message: {message}
    
    Respond with just the sentiment.""")
    
    chain = sentiment_prompt | llm
    result = chain.invoke({"message": last_message})
    sentiment = result.content.strip().lower()
    
    print(f"→ Sentiment: {sentiment}")
    return {"sentiment": sentiment}


# Bind tools to agent
llm_with_tools = llm.bind_tools(tools)


def agent_node(state: SupportState) -> dict:
    """Main agent that handles customer requests."""
    
    system_prompt = f"""You are a helpful customer support agent for an e-commerce company.

    Customer Context:
    - Intent: {state.get('intent', 'unknown')}
    - Sentiment: {state.get('sentiment', 'neutral')}
    
    Guidelines:
    1. Be friendly, professional, and empathetic
    2. Use the tools available to look up information
    3. If the customer is frustrated (negative sentiment), acknowledge their feelings first
    4. For complaints that can't be resolved, escalate to a human agent
    5. Always confirm actions before taking them (like initiating returns)
    6. Keep responses concise but helpful"""
    
    messages = [SystemMessage(content=system_prompt)] + list(state["messages"])
    response = llm_with_tools.invoke(messages)
    
    return {"messages": [response]}


def check_escalation(state: SupportState) -> dict:
    """Check if the conversation was escalated."""
    last_message = state["messages"][-1]
    
    if isinstance(last_message, AIMessage) and last_message.content:
        if "[ESCALATED]" in last_message.content:
            return {"escalated": True}
    
    return {"escalated": False}


# =============================================================================
# ROUTING: Determine flow through the graph
# =============================================================================

def should_analyze_or_respond(state: SupportState) -> Literal["analyze", "respond"]:
    """Decide if we need to analyze (first message) or respond directly."""
    # If we already have intent/sentiment, go to respond
    if state.get("intent") and state.get("sentiment"):
        return "respond"
    return "analyze"


def should_use_tools_or_end(state: SupportState) -> Literal["tools", "end"]:
    """Check if agent wants to use tools."""
    last_message = state["messages"][-1]
    
    if state.get("escalated"):
        return "end"
    
    if hasattr(last_message, "tool_calls") and last_message.tool_calls:
        return "tools"
    
    return "end"


# =============================================================================
# BUILD THE GRAPH
# =============================================================================

# Create tool node
tool_node = ToolNode(tools)

# Build the graph
builder = StateGraph(SupportState)

# Add nodes
builder.add_node("classify_intent", classify_intent)
builder.add_node("analyze_sentiment", analyze_sentiment)
builder.add_node("agent", agent_node)
builder.add_node("tools", tool_node)
builder.add_node("check_escalation", check_escalation)

# Add edges
builder.add_edge(START, "classify_intent")
builder.add_edge("classify_intent", "analyze_sentiment")
builder.add_edge("analyze_sentiment", "agent")
builder.add_conditional_edges(
    "agent",
    should_use_tools_or_end,
    {"tools": "tools", "end": "check_escalation"}
)
builder.add_edge("tools", "agent")  # After tools, back to agent
builder.add_edge("check_escalation", END)

# Add memory for conversation persistence
memory = MemorySaver()
support_bot = builder.compile(checkpointer=memory)


# =============================================================================
# CHAT INTERFACE
# =============================================================================

def chat(user_input: str, thread_id: str = "default") -> str:
    """Send a message to the support bot."""
    config = {"configurable": {"thread_id": thread_id}}
    
    result = support_bot.invoke(
        {"messages": [HumanMessage(content=user_input)]},
        config=config
    )
    
    # Get the last AI message
    for msg in reversed(result["messages"]):
        if isinstance(msg, AIMessage) and msg.content:
            return msg.content
    
    return "I apologize, but I'm having trouble processing your request."


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("Customer Support Chatbot Demo")
    print("=" * 60)
    
    # Simulate a conversation
    thread = "customer_12345"
    
    conversations = [
        "Hi, I need help with my order",
        "Yes, my order ID is ORD-001. Where is it?",
        "Great, thanks! Also, what's your return policy?",
        "I want to return order ORD-003",
    ]
    
    for user_msg in conversations:
        print(f"\n👤 Customer: {user_msg}")
        response = chat(user_msg, thread)
        print(f"🤖 Support: {response}")
    
    print("\n" + "=" * 60)
    print("Simulating an angry customer...")
    print("=" * 60)
    
    angry_thread = "angry_customer"
    
    print("\n👤 Customer: THIS IS RIDICULOUS! I've been waiting 3 weeks for my order!")
    response = chat("THIS IS RIDICULOUS! I've been waiting 3 weeks for my order and nobody is helping me!", angry_thread)
    print(f"🤖 Support: {response}")
