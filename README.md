# LangChain & LangGraph AI Tutorials

A collection of hands-on tutorials and a full-stack demo application for building AI agents with LangChain and LangGraph.

## Projects

### 1. LangChain/LangGraph Tutorial Series
**Location:** `langchain-langraph-tutorial/`

Progressive tutorials covering:
- **01** - LangChain Basics: Prompts, chains, and LLM interactions
- **02** - Tools & Agents: Building agents with custom tools
- **03** - LangGraph Basics: State machines and graph-based workflows
- **04** - LangGraph Advanced: Complex multi-step agent patterns
- **05** - Practical Customer Support: Production-ready agent implementation

### 2. Customer Support Chatbot Demo
**Location:** `customer-support-demo/`

A full-stack AI customer support application with real-time streaming.

**Features:**
- Real-time token streaming from LLM to UI
- Tool execution with expandable output display
- Thinking/reasoning indicator
- Markdown rendering with tables, code blocks, and lists
- WebSocket-based communication

**Tech Stack:**
| Component | Technology |
|-----------|------------|
| Frontend | React, TypeScript, Vite, Tailwind CSS |
| Backend | Python, FastAPI, WebSockets |
| AI | LangChain, LangGraph, Claude (Anthropic) |

**Tools Available:**
- `lookup_order` - Retrieve order details
- `search_knowledge_base` - Search FAQ and policies
- `check_return_eligibility` - Verify return eligibility
- `escalate_to_human` - Transfer to human agent

## Quick Start

### Prerequisites
- Python 3.10+
- Node.js 18+
- Azure OpenAI or Anthropic API key

### Running the Customer Support Demo

1. **Backend:**
   ```bash
   cd customer-support-demo/backend
   pip install -r requirements.txt
   # Create .env with your API keys (see .env.example)
   uvicorn main:app --reload --port 8000
   ```

2. **Frontend:**
   ```bash
   cd customer-support-demo/frontend
   npm install
   npm run dev
   ```

3. Open http://localhost:5173

### Running Tutorials
```bash
cd langchain-langraph-tutorial
pip install -r requirements.txt
# Create .env with your API keys
python 01_langchain_basics.py
```

## License

MIT
