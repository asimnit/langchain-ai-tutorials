"""
LangChain Basics Tutorial
=========================
This file covers the fundamental concepts of LangChain.
Using Claude on Azure AI Foundry as the LLM provider.
"""

# First, install required packages:
# pip install langchain langchain-anthropic python-dotenv

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# =============================================================================
# 1. CHAT MODELS - The core of LangChain
# =============================================================================

from langchain_anthropic import ChatAnthropic

# Initialize the model - Claude on Azure AI Foundry
llm = ChatAnthropic(
    model=os.getenv("AZURE_ANTHROPIC_MODEL", "claude-opus-4-6"),
    anthropic_api_url=os.getenv("AZURE_ANTHROPIC_ENDPOINT"),
    api_key=os.getenv("AZURE_ANTHROPIC_API_KEY"),
    max_tokens=1024,
    temperature=0.7,  # Controls randomness (0=deterministic, 1=creative)
)

# Simple invocation
response = llm.invoke("What is Python?")
print("Simple Response:", response.content)


# =============================================================================
# 2. PROMPT TEMPLATES - Reusable prompt structures
# =============================================================================

from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

# Basic prompt template
simple_prompt = ChatPromptTemplate.from_template(
    "You are a helpful assistant. Explain {topic} in simple terms for a {audience}."
)

# Create the prompt with variables
formatted_prompt = simple_prompt.format(topic="machine learning", audience="5-year-old")
print("\nFormatted Prompt:", formatted_prompt)

# More complex prompt with system and human messages
complex_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a {role} who speaks in a {style} manner."),
    ("human", "{question}")
])

# =============================================================================
# 3. CHAINS - Connecting components together
# =============================================================================

from langchain_core.output_parsers import StrOutputParser

# Create a chain using the | operator (LCEL - LangChain Expression Language)
chain = simple_prompt | llm | StrOutputParser()

# Run the chain
result = chain.invoke({
    "topic": "blockchain",
    "audience": "teenager"
})
print("\nChain Result:", result)


# =============================================================================
# 4. OUTPUT PARSERS - Structured output
# =============================================================================

from langchain_core.output_parsers import JsonOutputParser
from pydantic import BaseModel, Field

# Define the structure you want
class MovieRecommendation(BaseModel):
    title: str = Field(description="The movie title")
    year: int = Field(description="Release year")
    genre: str = Field(description="Primary genre")
    reason: str = Field(description="Why this movie is recommended")

# Create a parser
parser = JsonOutputParser(pydantic_object=MovieRecommendation)

# Create a prompt that includes format instructions
movie_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a movie expert. Always respond in JSON format."),
    ("human", "Recommend a {genre} movie. {format_instructions}")
])

# Create the chain with JSON output
movie_chain = movie_prompt | llm | parser

result = movie_chain.invoke({
    "genre": "sci-fi",
    "format_instructions": parser.get_format_instructions()
})
print("\nStructured Output:", result)


# =============================================================================
# 5. CONVERSATION MEMORY - Maintaining chat history
# =============================================================================

from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

# Manual message history
messages = [
    SystemMessage(content="You are a helpful coding assistant."),
    HumanMessage(content="My name is Alice and I'm learning Python."),
    AIMessage(content="Hello Alice! Great to meet you. Python is an excellent choice..."),
    HumanMessage(content="What's my name and what am I learning?")
]

# The model remembers the context
response = llm.invoke(messages)
print("\nWith Memory:", response.content)


# Using MessagesPlaceholder for dynamic history
prompt_with_history = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant."),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}")
])

chain_with_history = prompt_with_history | llm | StrOutputParser()

# Simulate a conversation
chat_history = []

def chat(user_input):
    response = chain_with_history.invoke({
        "chat_history": chat_history,
        "input": user_input
    })
    # Update history
    chat_history.append(HumanMessage(content=user_input))
    chat_history.append(AIMessage(content=response))
    return response

# Test the conversation
print("\n--- Conversation with Memory ---")
print("User: Hi, I'm Bob")
print("AI:", chat("Hi, I'm Bob"))
print("\nUser: What's my name?")
print("AI:", chat("What's my name?"))


# =============================================================================
# 6. STREAMING - Real-time output
# =============================================================================

print("\n--- Streaming Response ---")
for chunk in llm.stream("Tell me a short joke"):
    print(chunk.content, end="", flush=True)
print()


# =============================================================================
# 7. BATCH PROCESSING - Multiple inputs at once
# =============================================================================

simple_chain = ChatPromptTemplate.from_template("What is the capital of {country}?") | llm | StrOutputParser()

# Process multiple inputs
results = simple_chain.batch([
    {"country": "France"},
    {"country": "Japan"},
    {"country": "Brazil"}
])

print("\n--- Batch Results ---")
for r in results:
    print(r)
