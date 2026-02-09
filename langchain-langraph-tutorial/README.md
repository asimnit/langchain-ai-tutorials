# LangChain & LangGraph Tutorial

A hands-on tutorial for learning LangChain and LangGraph with practical code examples.

## 📁 Tutorial Structure

| File | Topic | What You'll Learn |
|------|-------|-------------------|
| `01_langchain_basics.py` | LangChain Fundamentals | Chat models, prompts, chains, memory, output parsers |
| `02_langchain_tools_agents.py` | Tools & Agents | Creating tools, binding to LLMs, agent execution loop |
| `03_langgraph_basics.py` | LangGraph Fundamentals | State, nodes, edges, conditional routing, loops |
| `04_langgraph_advanced.py` | Advanced LangGraph | ReAct agents, persistence, multi-agent systems |
| `05_practical_customer_support.py` | Real-World Example | Complete customer support chatbot |

## 🚀 Quick Start

```bash
# 1. Create virtual environment
python -m venv venv
venv\Scripts\activate  # Windows
source venv/bin/activate  # Mac/Linux

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set up environment variables
copy .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 4. Run the tutorials
python 01_langchain_basics.py
```

## 📚 Learning Path

### Beginner: Start with LangChain Basics
```
01_langchain_basics.py → Learn the building blocks
```
- How to use LLMs
- Prompt templates
- Chains (LCEL)
- Memory/conversation history

### Intermediate: Add Tools & Agents
```
02_langchain_tools_agents.py → Make LLMs take actions
```
- Create custom tools
- Tool binding
- Agent execution patterns
- RAG retrieval

### Advanced: LangGraph Workflows
```
03_langgraph_basics.py → Structured workflows
04_langgraph_advanced.py → Production patterns
```
- Graph-based architecture
- Conditional routing
- Loops and cycles
- Multi-agent systems
- Persistence

### Real-World Application
```
05_practical_customer_support.py → Put it all together
```
- Intent classification
- Sentiment analysis
- Knowledge base (RAG)
- Tool usage
- Escalation workflow

## 🔑 Key Concepts

### LangChain
```
┌─────────────────────────────────────────────────────┐
│                    LangChain                        │
├─────────────────────────────────────────────────────┤
│  Prompt → LLM → Output Parser                       │
│     ↓       ↓        ↓                              │
│  Template  Model   Structured Data                  │
│                                                     │
│  + Memory (conversation history)                    │
│  + Tools (functions LLM can call)                   │
│  + Retrievers (RAG / document lookup)               │
└─────────────────────────────────────────────────────┘
```

### LangGraph
```
┌─────────────────────────────────────────────────────┐
│                    LangGraph                        │
├─────────────────────────────────────────────────────┤
│                                                     │
│    START → [Node A] → [Node B] → END               │
│                 ↓                                   │
│            [Node C] ←──────────┘                   │
│                 ↑              (conditional edge)   │
│                 └──────────────┘ (loop!)           │
│                                                     │
│  State flows through nodes, edges control flow      │
│  Supports: loops, branching, persistence            │
└─────────────────────────────────────────────────────┘
```

## 💡 When to Use What

| Use Case | LangChain | LangGraph |
|----------|-----------|-----------|
| Simple Q&A | ✅ | ❌ |
| RAG Pipeline | ✅ | ❌ |
| Single Tool Agent | ✅ | ✅ |
| Multi-step Workflow | ❌ | ✅ |
| Loops/Retries | ❌ | ✅ |
| Multi-Agent System | ❌ | ✅ |
| Human-in-the-loop | ⚠️ Basic | ✅ |
| Persistence/Memory | ⚠️ Basic | ✅ |

## 📖 Additional Resources

- [LangChain Documentation](https://python.langchain.com/docs/)
- [LangGraph Documentation](https://langchain-ai.github.io/langgraph/)
- [LangSmith](https://smith.langchain.com/) - Debugging & tracing
- [LangChain Hub](https://smith.langchain.com/hub) - Prompt templates

## ⚠️ Notes

- All examples use OpenAI models (GPT-4). Modify for other providers.
- Some examples use simulated data. Replace with real APIs in production.
- For production, use `SqliteSaver` or `PostgresSaver` instead of `MemorySaver`.
