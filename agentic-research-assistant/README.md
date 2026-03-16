# Multi-Agent Research Assistant

An autonomous internet research system powered by open-source LLMs and multi-agent orchestration. Input a research topic and receive structured, cited results — no paid APIs required.

## System Architecture

```
User Question
     │
     ▼
┌─────────────────────┐
│  Query Understanding │  ← Parses intent, extracts topics & search queries
│       Agent          │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Research Planner    │  ← Creates research plan, refines queries
│       Agent          │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│    Search Agent      │  ← Executes DuckDuckGo searches
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Web Scraper Agent   │  ← Downloads & extracts page content
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Information         │  ← LLM extracts structured insights
│  Extraction Agent    │
└─────────┬───────────┘
          │
          ▼
┌─────────────────────┐
│  Research Synthesizer│  ← Deduplicates, organizes, generates table
│       Agent          │
└─────────┬───────────┘
          │
          ▼
   Structured Table
   with Citations
```

The pipeline is orchestrated by **LangGraph**, which manages state flow between agents and handles errors gracefully.

## Features

- **Natural language research prompts** — ask any research question
- **Autonomous internet research** — searches, scrapes, and extracts without manual intervention
- **Multi-agent architecture** — six specialized agents, each with a distinct role
- **Structured output** — results presented as a table with citations
- **Multiple output formats** — table, bullet points
- **Follow-up questions** — conversation memory enables iterative refinement
- **Open-source LLMs** — runs locally with HuggingFace models, no paid APIs
- **Optional quantization** — 4-bit quantization for lower VRAM requirements
- **Gradio web UI** — clean browser-based interface

## Supported Models

| Model | Parameters | Notes |
|-------|-----------|-------|
| `microsoft/Phi-3-mini-4k-instruct` | 3.8B | Default — fast, low VRAM |
| `mistralai/Mistral-7B-Instruct-v0.3` | 7B | Good balance of quality and speed |
| `meta-llama/Meta-Llama-3-8B-Instruct` | 8B | Best quality — requires HF token and license acceptance |

## Installation

### Prerequisites

- Python 3.10+
- CUDA-capable GPU with ≥8 GB VRAM (or use quantization for lower VRAM)
- Git

### Setup

```bash
git clone https://github.com/virendrakumar93/Web_Research.git
cd Web_Research/agentic-research-assistant
pip install -r requirements.txt
```

### HuggingFace Token (required for gated models)

Some models (e.g., Llama 3) require accepting a license on HuggingFace and setting an API token.

1. Create a free account at [huggingface.co](https://huggingface.co)
2. Accept the model license on the model page
3. Generate a token at [huggingface.co/settings/tokens](https://huggingface.co/settings/tokens)
4. Set the token:

```bash
export HF_TOKEN="hf_your_token_here"
```

## Running

### Basic usage (default model: Phi-3-mini)

```bash
python app.py
```

### Select a specific model

```bash
python app.py --model mistralai/Mistral-7B-Instruct-v0.3
```

### Enable 4-bit quantization (lower VRAM)

```bash
python app.py --quantize
```

### Create a public share link

```bash
python app.py --share
```

### Custom port

```bash
python app.py --port 8080
```

Then open **http://localhost:7860** in your browser (or the port you specified).

## Configuration

Edit `config/config.yaml` to customize:

```yaml
llm:
  model_name: "microsoft/Phi-3-mini-4k-instruct"
  temperature: 0.7
  max_tokens: 1024
  quantize: false

search:
  max_results: 5

scraper:
  timeout: 15
  max_content_length: 10000

research:
  max_queries: 3
  max_urls_per_query: 3
  max_sources: 10
```

## Example Prompts

- "Compare the top 5 open-source LLMs released in 2024 by performance, parameter count, and license"
- "What are the latest advances in solid-state battery technology?"
- "Summarize the pros and cons of Rust vs Go for backend development"
- "What are the most effective strategies for reducing cloud computing costs?"
- "Research the current state of quantum computing hardware vendors"

## Project Structure

```
agentic-research-assistant/
├── app.py                          # Entry point
├── requirements.txt                # Dependencies
├── README.md                       # This file
├── config/
│   └── config.yaml                 # Configuration
├── llm/
│   ├── model_loader.py             # HuggingFace model loading & generation
│   └── prompts.py                  # All prompt templates
├── agents/
│   ├── query_agent.py              # Query Understanding Agent
│   ├── planner_agent.py            # Research Planner Agent
│   ├── search_agent.py             # Search Agent (DuckDuckGo)
│   ├── scraper_agent.py            # Web Scraper Agent
│   ├── extraction_agent.py         # Information Extraction Agent
│   └── synthesis_agent.py          # Research Synthesizer Agent
├── tools/
│   ├── web_search.py               # DuckDuckGo search wrapper
│   ├── scraper.py                  # Web scraping (trafilatura + BS4)
│   └── text_cleaner.py             # Text cleaning & chunking
├── workflows/
│   └── research_graph.py           # LangGraph workflow orchestration
├── memory/
│   └── conversation_memory.py      # Multi-turn conversation memory
├── ui/
│   └── gradio_app.py               # Gradio web interface
└── utils/
    └── logger.py                   # Logging utility
```

## Technology Stack

| Component | Technology |
|-----------|-----------|
| Language | Python 3.10+ |
| LLMs | HuggingFace Transformers + Accelerate |
| Quantization | bitsandbytes (optional) |
| Agent Orchestration | LangGraph |
| Web Search | duckduckgo-search |
| Web Scraping | requests + BeautifulSoup4 + trafilatura |
| UI | Gradio |
| Data | pandas |

## License

MIT
