# Zero-Budget Autonomous Web Pentesting Framework

> Fully autonomous, multi-agent web penetration testing powered by LangGraph — zero infrastructure cost, production-grade architecture.

---

## Overview

This framework orchestrates **19 specialized AI agents** in a deterministic graph to perform end-to-end web application penetration testing autonomously. It handles reconnaissance, crawling, vulnerability hypothesis generation, payload crafting, execution, validation, CVSS scoring, and report generation — all without human intervention.

Key design principles:

- **Zero-budget**: Built exclusively on free-tier and open API providers (Groq, Cerebras, OpenRouter free models, Gemini, Cloudflare Workers AI).
- **Resilient**: Every LLM call has 4–5 automatic fallbacks across providers. A single key expiring never breaks a session.
- **Safe**: Scope enforcement is a hard gate before every network action. The framework refuses any target outside the declared scope.
- **Stateful**: LangGraph manages the full session state — every agent reads from and writes to a typed `AppState`.

---

## Architecture

The graph is a single directed loop with one conditional router (the Orchestrator) and two deterministic bypass edges for the executor → parser pipelines.

```
START
  │
  ▼
scope_enforcer ──────────────────────────────────────────────────────────┐
  │                                                                       │
  ▼                                                                       │
orchestrator ◄──────── (all nodes loop back here) ──────────────────────┘
  │
  ├─► error_handler
  ├─► planner
  │     (sets next_phase)
  │
  ├─► recon_executor ──► recon_parser ──────────────────────► orchestrator
  ├─► crawler_executor ──► crawler_parser ──────────────────► orchestrator
  │
  ├─► web_filter ────────────────────────────────────────────► orchestrator
  ├─► hypothesis_analyzer ───────────────────────────────────► orchestrator
  ├─► knowledge_rag ─────────────────────────────────────────► orchestrator
  ├─► payload_generator ─────────────────────────────────────► orchestrator
  ├─► payload_optimizer ─────────────────────────────────────► orchestrator
  ├─► execution_sandbox ─────────────────────────────────────► orchestrator
  ├─► validator ─────────────────────────────────────────────► orchestrator
  ├─► cvss_engine ───────────────────────────────────────────► orchestrator
  ├─► business_impact ───────────────────────────────────────► orchestrator
  ├─► reporter ──────────────────────────────────────────────► orchestrator
  └─► memory_summarizer ─────────────────────────────────────► END
```

### Node Responsibilities

| Node | Purpose | LLM Chain |
|---|---|---|
| `scope_enforcer` | Validates target is within declared scope. Hard-fails outside scope. | — |
| `orchestrator` | Priority-based deterministic router. No LLM involved. | — |
| `planner` | Decides the next attack phase using chain-of-thought reasoning. | `deepseek_r1` |
| `recon_executor` | Runs Nmap + Subfinder subprocesses; writes `raw_recon_output`. | — |
| `recon_parser` | Parses Nmap/Subfinder output into structured `ReconData`. | `gemini_flash` |
| `crawler_executor` | Crawls target with Playwright; writes `raw_crawler_output`. | — |
| `crawler_parser` | Parses request log + HTML into structured `CrawlerData`. | `gemini_flash` |
| `web_filter` | Filters crawled endpoints to web-only attack surface. | `gemini_flash` |
| `hypothesis_analyzer` | Generates ranked vulnerability hypotheses from attack surface. | `deepseek_r1` |
| `knowledge_rag` | Retrieves relevant pentesting methodology from Qdrant vector DB. | `gemini_flash` |
| `payload_generator` | Crafts benign PoC payloads for each hypothesis. | `deepseek_v3` |
| `payload_optimizer` | Refines INCONCLUSIVE payloads for retry. | `deepseek_v3` |
| `execution_sandbox` | Executes payloads via httpx with rate limiting and concurrency control. | — |
| `validator` | Classifies execution results as TRUE_POSITIVE / FALSE_POSITIVE / INCONCLUSIVE. | `deepseek_r1` |
| `cvss_engine` | Computes CVSS v3.1 score for confirmed findings. | `deepseek_r1` |
| `business_impact` | Assesses business impact of confirmed findings. | `gemini_pro` |
| `reporter` | Generates the final Markdown pentest report. | `gemini_pro` |
| `memory_summarizer` | Compresses session state to prevent context overflow. | `gemini_pro` |
| `error_handler` | Handles errors, retries, and graceful degradation. | — |

### LLM Chains and Fallback Order

| Chain | Purpose | Provider Order |
|---|---|---|
| `deepseek_r1` | Reasoning (Planner, Validator, Hypothesis, CVSS) | Groq 70B → GitHub 70B → OpenRouter llama-3.3-70B → Cohere → Mistral |
| `deepseek_v3` | Payloads (Generator, Optimizer) | OpenRouter qwen3-coder → OpenRouter dolphin-mistral-24B → Groq 70B → GitHub 70B → OpenRouter nemotron-550B |
| `gemini_flash` | Parsing (Recon, Crawler, Web Filter, RAG) | Groq 8B → Cerebras 8B → Gemini 2.5 Flash → Cloudflare Workers AI |
| `gemini_pro` | Reporting (Reporter, Business Impact, Memory) | Gemini 2.5 Pro → Cohere → Mistral → GitHub 70B |

---

## Installation

### Requirements

- Linux (Ubuntu 22.04+ recommended)
- Python 3.12+
- `pip`

### 1. Clone and install

```bash
git clone https://github.com/your-org/msa_project.git
cd msa_project
pip install -e .
```

### 2. Install external recon tools

```bash
# Nmap
sudo apt-get install -y nmap

# Subfinder (requires Go 1.21+)
go install -v github.com/projectdiscovery/subfinder/v2/cmd/subfinder@latest
# Ensure $GOPATH/bin is on your $PATH:
export PATH="$PATH:$(go env GOPATH)/bin"

# Playwright (browser binary)
pip install playwright
playwright install chromium
# On headless servers, also run:
playwright install-deps chromium
```

All three tools are optional — the framework degrades gracefully if any are missing (missing tools produce empty raw output; the downstream parser emits a warning and the orchestrator continues).

### 3. Configure environment

```bash
cp .env.example .env
# Edit .env with your actual API keys (see Environment Variables below)
```

### 4. Create your scope file

```bash
cp scope.json.example scope.json
# Edit scope.json with the target's actual domain(s)
```

---

## Environment Variables

Copy `.env.example` to `.env` and fill in at least one primary key. Every provider is optional except at least one of `GEMINI_API_KEYS` or `GROQ_API_KEYS`.

```env
# PRIMARY — at least one required
GEMINI_API_KEYS=AIzaSy...          # https://aistudio.google.com/app/apikey
GROQ_API_KEYS=gsk_...              # https://console.groq.com/keys

# FALLBACK PROVIDERS — all optional but strongly recommended
CEREBRAS_API_KEYS=csk-...          # https://cloud.cerebras.ai
GITHUB_API_KEYS=ghp_...            # https://github.com/settings/tokens
OPENROUTER_API_KEYS=sk-or-v1-...   # https://openrouter.ai/keys
MISTRAL_API_KEYS=...               # https://console.mistral.ai/api-keys
COHERE_API_KEYS=...                # https://dashboard.cohere.com/api-keys
CLOUDFLARE_API_KEYS=cfut_...       # https://dash.cloudflare.com/profile/api-tokens
CLOUDFLARE_ACCOUNT_ID=...          # https://dash.cloudflare.com → right sidebar

# MODELS (defaults shown)
GEMINI_FLASH_MODEL=gemini-2.5-flash
GEMINI_PRO_MODEL=gemini-2.5-pro

# TUNING
EXECUTION_TIMEOUT_SECONDS=30
EXECUTION_MAX_CONCURRENT=5
EXECUTION_RATE_LIMIT_RPS=10
VALIDATION_CONFIDENCE_THRESHOLD=0.6
HYPOTHESIS_CONFIDENCE_THRESHOLD=0.4
LOG_LEVEL=INFO
SESSION_MAX_RETRIES=3
```

---

## scope.json

The scope file declares what the framework is authorized to test. It is validated at startup — the session is aborted if the target falls outside the declared scope.

```json
{
  "in_scope_domains": [
    "example.com",
    "api.example.com"
  ],
  "in_scope_cidrs": [],
  "out_of_scope_domains": [
    "admin.example.com"
  ],
  "out_of_scope_cidrs": [],
  "allowed_ports": [80, 443, 8080, 8443],
  "forbidden_paths": [
    "/admin",
    "/logout",
    "/api/v1/internal"
  ],
  "requires_auth": false,
  "auth_config": null,
  "max_requests_per_second": 10,
  "max_concurrent_requests": 5,
  "legal_acknowledged": true
}
```

`legal_acknowledged` must be `true`. Setting it to `false` will cause the `scope_enforcer` to abort immediately.

---

## CLI Usage

After `pip install -e .`, the `pentest` command is available globally.

```bash
# Basic usage
pentest --target https://example.com/

# Custom scope file
pentest --target https://example.com/ --scope /path/to/scope.json

# Limit the number of orchestrator cycles (default: 200)
pentest --target https://example.com/ --max-cycles 50

# Override log level
pentest --target https://example.com/ --log-level DEBUG

# Specify HTTP method
pentest --target https://example.com/api/login --method POST

# Full help
pentest --help
```

Reports are written to `data/reports/report_<session_id>.md` on completion.

---

## End-to-End Execution Flow

```
pentest --target https://target.com/ --scope scope.json
         │
         ▼
1.  Load scope.json → validate ScopeConfig
2.  Build AppState (session_id, target, scope, scope_verified=False)
3.  Compile 19-node LangGraph → invoke graph

GRAPH EXECUTION:
─────────────────────────────────────────────────────────────────
 scope_enforcer   Verifies target.com ∈ in_scope_domains
       │
 orchestrator     scope_verified=False → route to scope_enforcer (first pass)
                  scope_verified=True  → route to planner
       │
 planner          "raw_recon_output is empty → recon_executor"
       │
 recon_executor   Nmap top-1000 ports + Subfinder subdomains
       │           → writes raw_recon_output
 recon_parser     LLM parses XML/text → ReconData (open ports, services)
       │
 orchestrator     planner → "crawler_executor"
       │
 crawler_executor  Playwright BFS crawl (depth=3, max 30 pages)
       │            → writes raw_crawler_output
 crawler_parser   LLM parses request log + HTML → CrawlerData (endpoints, forms)
       │
 web_filter       Filters to web-only attack surface
       │
 hypothesis_analyzer  Generates ranked VulnHypotheses (SQLi, XSS, IDOR…)
       │
 knowledge_rag    Retrieves relevant methodology from Qdrant
       │
 payload_generator  Crafts PoC payloads per hypothesis
       │             → sets active_payload_id
 execution_sandbox   Sends payloads via httpx (rate-limited, scoped)
       │
 validator        TRUE_POSITIVE / FALSE_POSITIVE / INCONCLUSIVE
       │
 [if INCONCLUSIVE] → payload_optimizer → execution_sandbox → validator
       │
 [if TRUE_POSITIVE] → cvss_engine → business_impact
       │
 reporter         Generates Markdown report with all confirmed findings
       │
 memory_summarizer Compresses state, clears raw buffers
       │
       ▼
data/reports/report_<session_id>.md
─────────────────────────────────────────────────────────────────
```

---

## Legal Notice

**This framework is designed exclusively for authorized penetration testing.**

- You must have **explicit written permission** from the system owner before running any scan or test.
- Setting `"legal_acknowledged": true` in `scope.json` is your declaration that you have obtained the necessary authorization.
- The `scope_enforcer` node will abort any session where the target falls outside the declared `in_scope_domains` or `in_scope_cidrs`.
- The authors accept no liability for misuse. Unauthorized testing is illegal in most jurisdictions.

Use responsibly and ethically.

---

## Project Structure

```
msa_project/
├── src/
│   ├── main.py                  # CLI entry point
│   ├── graph/
│   │   └── builder.py           # LangGraph compilation
│   ├── agents/                  # 19 agent nodes
│   │   ├── scope_enforcer.py
│   │   ├── orchestrator.py
│   │   ├── planner.py
│   │   ├── recon_executor.py
│   │   ├── recon_parser.py
│   │   ├── crawler_executor.py
│   │   ├── crawler_parser.py
│   │   ├── web_filter.py
│   │   ├── hypothesis_analyzer.py
│   │   ├── knowledge_rag.py
│   │   ├── payload_generator.py
│   │   ├── payload_optimizer.py
│   │   ├── execution_sandbox.py
│   │   ├── validator.py
│   │   ├── cvss_engine.py
│   │   ├── business_impact.py
│   │   ├── reporter.py
│   │   ├── memory_summarizer.py
│   │   └── error_handler.py
│   └── shared/
│       ├── config.py            # Pydantic settings
│       ├── llm.py               # Multi-provider fallback chains
│       ├── schemas.py           # All Pydantic models
│       ├── state.py             # AppState TypedDict
│       ├── logging.py           # Structured logging
│       └── exceptions.py       # Framework exceptions
├── data/
│   ├── knowledge_base/          # Qdrant source documents
│   ├── qdrant/                  # Qdrant persistence (local)
│   └── reports/                 # Generated pentest reports
├── scope.json                   # Your scope (gitignore this)
├── scope.json.example
├── .env                         # Your API keys (gitignore this)
├── .env.example
└── pyproject.toml
```

---

*Built with [LangGraph](https://github.com/langchain-ai/langgraph) · Runs on free-tier APIs · Linux only*