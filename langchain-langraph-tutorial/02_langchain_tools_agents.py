"""
LangChain Tools & Agents Tutorial
=================================
This file shows how to create tools and agents that can use them.
Using Claude on Azure AI Foundry as the LLM provider.
"""

import os
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.tools import tool
from langchain_core.messages import HumanMessage

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
# 1. CREATING TOOLS - Functions the LLM can use
# =============================================================================

# Method 1: Using the @tool decorator (simplest)
@tool
def get_weather(city: str) -> str:
    """Get the current weather for a city. Use this when asked about weather."""
    # In real app, call a weather API
    weather_data = {
        "New York": "Sunny, 72°F",
        "London": "Cloudy, 55°F",
        "Tokyo": "Rainy, 65°F"
    }
    return weather_data.get(city, f"Weather data not available for {city}")


@tool
def calculate(expression: str) -> str:
    """Calculate a mathematical expression. Use for any math operations."""
    try:
        result = eval(expression)  # Note: Use safer evaluation in production
        return str(result)
    except Exception as e:
        return f"Error: {e}"


@tool
def search_products(query: str, max_price: float = 100.0) -> str:
    """Search for products by name. Optionally filter by maximum price."""
    # Simulated product database
    products = [
        {"name": "Laptop", "price": 999.99},
        {"name": "Mouse", "price": 29.99},
        {"name": "Keyboard", "price": 79.99},
        {"name": "Monitor", "price": 299.99},
    ]
    results = [p for p in products if query.lower() in p["name"].lower() and p["price"] <= max_price]
    return str(results) if results else "No products found"


# =============================================================================
# 2. BINDING TOOLS TO THE MODEL
# =============================================================================

# Give the LLM access to tools
tools = [get_weather, calculate, search_products]
llm_with_tools = llm.bind_tools(tools)

# The model will now decide when to use tools
response = llm_with_tools.invoke("What's the weather in Tokyo?")
print("Tool Call Response:", response.tool_calls)


# =============================================================================
# 3. TOOL EXECUTION LOOP - Complete agent pattern
# =============================================================================

from langchain_core.messages import AIMessage, ToolMessage

def run_agent(user_input: str) -> str:
    """Run a simple agent that can use tools."""
    messages = [HumanMessage(content=user_input)]
    
    while True:
        # Get model response
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        
        # Check if model wants to use tools
        if not response.tool_calls:
            # No tool calls, return the response
            return response.content
        
        # Execute each tool call
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            tool_args = tool_call["args"]
            
            print(f"  → Calling tool: {tool_name} with args: {tool_args}")
            
            # Find and execute the tool
            tool_map = {t.name: t for t in tools}
            tool_result = tool_map[tool_name].invoke(tool_args)
            
            # Add tool result to messages
            messages.append(ToolMessage(
                content=str(tool_result),
                tool_call_id=tool_call["id"]
            ))
    
    return response.content


# Test the agent
print("\n--- Agent Demo ---")
print("Q: What's the weather in London?")
print("A:", run_agent("What's the weather in London?"))

print("\nQ: What is 25 * 47 + 123?")
print("A:", run_agent("What is 25 * 47 + 123?"))

print("\nQ: Find me products under $50")
print("A:", run_agent("Find me products under $50"))


# =============================================================================
# 4. STRUCTURED TOOLS - More complex tool definitions
# =============================================================================

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field


# Define input schema
class EmailInput(BaseModel):
    to: str = Field(description="Email recipient address")
    subject: str = Field(description="Email subject line")
    body: str = Field(description="Email body content")


def send_email_func(to: str, subject: str, body: str) -> str:
    """Send an email to the specified recipient."""
    # In production, actually send the email
    return f"Email sent to {to} with subject: {subject}"


# Create a structured tool
send_email = StructuredTool.from_function(
    func=send_email_func,
    name="send_email",
    description="Send an email to someone",
    args_schema=EmailInput
)


# =============================================================================
# 5. RETRIEVAL TOOLS - RAG Pattern
# =============================================================================
# Using Azure OpenAI Embeddings for vector search

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import AzureOpenAIEmbeddings

# Create sample documents
documents = [
    Document(page_content="LangChain is a framework for building LLM applications.", metadata={"source": "docs"}),
    Document(page_content="LangGraph extends LangChain with graph-based workflows.", metadata={"source": "docs"}),
    Document(page_content="Agents can use tools to interact with external systems.", metadata={"source": "tutorial"}),
    Document(page_content="Vector stores enable semantic search over documents.", metadata={"source": "tutorial"}),
]

# Azure OpenAI Embeddings
embeddings = AzureOpenAIEmbeddings(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    azure_deployment=os.getenv("AZURE_OPENAI_EMBEDDING_DEPLOYMENT", "text-embedding-3-small"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-01"),
)

# Create vector store
vectorstore = InMemoryVectorStore.from_documents(documents, embeddings)
retriever = vectorstore.as_retriever(search_kwargs={"k": 2})


@tool
def search_knowledge_base(query: str) -> str:
    """Search the knowledge base for relevant information."""
    docs = retriever.invoke(query)
    if docs:
        return "\n".join([doc.page_content for doc in docs])
    return "No relevant information found."


# Now the agent can search documents!
tools_with_retrieval = [search_knowledge_base, calculate]
llm_with_retrieval = llm.bind_tools(tools_with_retrieval)


# =============================================================================
# 6. HUMAN-IN-THE-LOOP - Asking for confirmation
# =============================================================================

@tool
def delete_file(filename: str) -> str:
    """Delete a file from the system. DANGEROUS - requires confirmation."""
    # In production, actually delete
    return f"File {filename} would be deleted"


# Add delete_file to tools for the confirmation agent
tools_with_delete = [get_weather, calculate, search_products, delete_file]
llm_with_delete_tools = llm.bind_tools(tools_with_delete)


def run_agent_with_confirmation(user_input: str) -> str:
    """Agent that asks for confirmation on dangerous operations."""
    messages = [HumanMessage(content=user_input)]
    dangerous_tools = {"delete_file"}  # Tools that need confirmation
    
    while True:
        response = llm_with_delete_tools.invoke(messages)
        messages.append(response)
        
        if not response.tool_calls:
            return response.content
        
        for tool_call in response.tool_calls:
            tool_name = tool_call["name"]
            
            # Check if this is a dangerous operation
            if tool_name in dangerous_tools:
                print(f"\n⚠️  The agent wants to call: {tool_name}")
                print(f"   Arguments: {tool_call['args']}")
                confirm = input("   Allow this operation? (yes/no): ")
                
                if confirm.lower() != "yes":
                    return "Operation cancelled by user."
            
            # Execute the tool
            tool_map = {t.name: t for t in tools_with_delete}
            if tool_name in tool_map:
                tool_result = tool_map[tool_name].invoke(tool_call["args"])
                messages.append(ToolMessage(
                    content=str(tool_result),
                    tool_call_id=tool_call["id"]
                ))


# =============================================================================
# 7. PARALLEL TOOL CALLS - Multiple tools at once
# =============================================================================

@tool
def get_stock_price(symbol: str) -> str:
    """Get the current stock price for a symbol."""
    prices = {"AAPL": 150.25, "GOOGL": 140.50, "MSFT": 380.00}
    return f"{symbol}: ${prices.get(symbol, 'N/A')}"


@tool
def get_company_info(symbol: str) -> str:
    """Get company information for a stock symbol."""
    info = {
        "AAPL": "Apple Inc. - Technology company",
        "GOOGL": "Alphabet Inc. - Technology conglomerate",
        "MSFT": "Microsoft Corp. - Software company"
    }
    return info.get(symbol, "Company not found")


# The LLM can call multiple tools in parallel
parallel_tools = [get_stock_price, get_company_info]
llm_parallel = llm.bind_tools(parallel_tools)

# This might result in parallel tool calls
response = llm_parallel.invoke("Tell me about Apple stock - both price and company info")
print("\n--- Parallel Tool Calls ---")
print("Tool calls:", len(response.tool_calls))
for tc in response.tool_calls:
    print(f"  - {tc['name']}: {tc['args']}")
