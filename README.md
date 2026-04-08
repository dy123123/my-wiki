# llm-wiki

An LLM-maintained local markdown wiki. Add documents, let the LLM analyze them, and build a structured knowledge base you can query, search, and audit — all stored as plain git-friendly markdown files.

Inspired by Andrej Karpathy's LLM Wiki pattern.

## Features

- **Any OpenAI-compatible LLM** — OpenAI, Ollama, LM Studio, vLLM, Anthropic-compatible proxies
- **markitdown** — automatic conversion of PDF, Word, Excel, PowerPoint, HTML, images, audio, and more
- **Structured extraction** — entities, concepts, topics, and tags extracted via JSON-mode LLM calls
- **Incremental ingestion** — re-ingest a source to update existing wiki pages without duplication
- **Full-text search** — keyword search across all wiki pages
- **RAG retrieval** — chunk + embed sources; `ask` searches wiki pages first, then retrieves precise chunks from raw documents (register addresses, specs, exact values)
- **Reranking** — optional cross-encoder reranker (Qwen3-Reranker, etc.) for better chunk ranking
- **Lint** — detect orphan pages, dead links, duplicate slugs, missing sections, empty pages
- **Q&A** — ask questions answered with citations from the wiki + RAG chunks; optionally save to `analyses/`
- **MCP server** — `llm-wiki mcp` exposes `wiki_ask`, `wiki_search`, `wiki_page`, `wiki_status` tools for OpenCode, Claude Desktop, Cursor, Cline, and other MCP clients
- **Dry-run support** — preview any destructive operation before running it
- **Immutable raw files** — source files are write-protected once added

## Installation

```bash
# Minimal install (no markitdown — plain text sources only)
pip install .

# Recommended: basic normalization (text, HTML, CSV, XML)
pip install ".[normalize]"

# Full normalization: + PDF, Word, Excel, PowerPoint, images, audio
pip install ".[normalize-full]"

# With RAG support (numpy for embedding storage)
pip install ".[rag]"

# With MCP server for OpenCode, Claude Desktop, Cursor, Cline
pip install ".[mcp]"

# Everything
pip install ".[all]"

# Development
pip install -e ".[normalize,dev]"
```

**Requirements:** Python 3.10+

> **Why is markitdown optional?** `markitdown[all]` pulls in heavy ML libraries
> (speech recognition, Azure cognitive services, image processing) that take a long
> time to install. The core tool works fine without it — only the `normalize` command
> needs it, and plain `.txt`/`.md` files are handled natively.

## Quick Start

```bash
# 1. Install (with normalization support)
pip install ".[normalize]"      # basic
pip install ".[normalize-full]" # + PDF, Office, images, audio

# 2. Copy and fill in your API key
cp .env.example .env
$EDITOR .env

# 3. Initialize a vault in ./vault/
llm-wiki init

# 4. Verify LLM connectivity
llm-wiki llm ping

# 5. Add a source document
llm-wiki add path/to/paper.pdf --tag ai --tag nlp

# 6. Convert to markdown
llm-wiki normalize <source-id>

# 7. Analyze and build wiki pages
llm-wiki ingest <source-id>

# 7. Ask questions
llm-wiki ask "What is self-attention and why is it useful?"

# 8. Search
llm-wiki search "transformer attention mechanism"

# 9. Check health
llm-wiki status
llm-wiki lint
```

## Commands

### `llm-wiki init [--vault PATH] [--dry-run]`
Initialize a new vault. Creates the full directory structure and seeds schema docs.

### `llm-wiki config show`
Display current configuration (API key is masked).

### `llm-wiki config validate`
Validate configuration; exits with code 1 if anything is missing.

### `llm-wiki llm ping`
Send a test request to the LLM backend and verify it responds.

### `llm-wiki add <path> [--tag TAG]... [--dry-run]`
Copy a file into `vault/raw/`, generate a source ID, and write a metadata sidecar.
- Tags can be repeated: `--tag ai --tag nlp`
- The raw file is made read-only (`chmod 444`) — treat it as immutable
- Re-adding an identical file is idempotent

### `llm-wiki normalize [source-id] [--all] [--dry-run]`
Convert a raw source to markdown in `vault/normalized/`.
- Uses **markitdown** for PDF, Office, HTML, images, audio, and more
- Falls back to plain text for unsupported formats

### `llm-wiki ingest [source-id] [--latest] [--all] [--dry-run]`
Analyze the normalized content with the LLM and build/update wiki pages:
1. Extracts structured analysis (entities, concepts, topics, key points)
2. Generates a source summary page in `wiki/sources/`
3. Creates or updates entity, concept, and topic pages
4. Updates `wiki/index.md`
5. Appends to `wiki/log.md`

### `llm-wiki process [source-id] [--latest] [--all] [--dry-run]`
Normalize, ingest, and embed a source in one step (shortcut for running all three commands sequentially).
Auto-embeds if `LLM_WIKI_EMBED_MODEL` is set.

### `llm-wiki embed [source-id] [--all] [--force] [--dry-run]`
Chunk and embed normalized sources for RAG retrieval.
Requires `LLM_WIKI_EMBED_MODEL` to be set. Called automatically by `process` when configured.

### `llm-wiki ask "<question>" [--save] [--no-rag]`
Answer a question from the wiki + RAG:
1. Searches wiki pages by keyword relevance (structural knowledge)
2. If `LLM_WIKI_EMBED_MODEL` is set, retrieves relevant chunks from raw source documents (precise values, specs)
3. Optionally reranks chunks with `LLM_WIKI_RERANK_MODEL`
4. Returns an answer with citations and confidence level
- `--save` writes the answer to `wiki/analyses/`
- `--no-rag` skips RAG and uses only wiki pages

### `llm-wiki search "<query>"`
Full-text keyword search across all wiki pages, with scored results and inline snippets.

### `llm-wiki lint [--fix] [--dry-run]`
Detect structural issues:
| Category | Severity | Description |
|----------|----------|-------------|
| `orphan` | warning | Page not referenced in `index.md` |
| `dead_link` | error | Relative link points to missing file |
| `duplicate` | warning | Entity/concept with near-identical slug |
| `missing_section` | warning | Required section (e.g. `## Sources`) absent |
| `empty_page` | info | Page with very little content |

`--fix` auto-corrects simple issues (adds orphans to index). Exit code 1 if errors found.

### `llm-wiki status`
Show vault health: source counts, normalization/ingestion status, recent log entries, and pending actions.

### `llm-wiki log tail [-n N]`
Show the N most recent log entries (newest first). Default: 20.

### `llm-wiki mcp [--http] [--host HOST] [--port PORT] [--token TOKEN]`
Start an MCP server that exposes the wiki to AI coding assistants and chat clients.

**Stdio mode** (local — for OpenCode, Claude Desktop, Cursor, Cline on the same machine):
```bash
pip install ".[mcp]"
llm-wiki mcp
```

**HTTP/SSE mode** (remote access from another machine):
```bash
llm-wiki mcp --http --port 8080 --token mysecrettoken
# → SSE endpoint: http://<host>:8080/sse
# → Health check: http://<host>:8080/health
```

| MCP Tool | Description |
|----------|-------------|
| `wiki_ask` | Answer a question using wiki pages + RAG chunks |
| `wiki_search` | Full-text keyword search across wiki pages |
| `wiki_page` | Read a specific wiki page by path |
| `wiki_status` | Show vault stats (sources, pages, embeddings) |

#### OpenCode / Claude Desktop config (stdio)

```json
{
  "mcpServers": {
    "my-wiki": {
      "command": "llm-wiki",
      "args": ["mcp"]
    }
  }
}
```

#### Remote HTTP/SSE config

```json
{
  "mcpServers": {
    "my-wiki": {
      "type": "sse",
      "url": "http://<server>:8080/sse",
      "headers": {"Authorization": "Bearer mysecrettoken"}
    }
  }
}
```

## Configuration

All settings are read from environment variables (prefixed `LLM_WIKI_`) or a `.env` file:

| Variable | Default | Description |
|----------|---------|-------------|
| `LLM_WIKI_LLM_BASE_URL` | `https://api.openai.com/v1` | API base URL |
| `LLM_WIKI_LLM_API_KEY` | _(required)_ | API key |
| `LLM_WIKI_LLM_MODEL` | `gpt-4o-mini` | Model name |
| `LLM_WIKI_LLM_TEMPERATURE` | `0.2` | Sampling temperature |
| `LLM_WIKI_LLM_MAX_TOKENS` | `4096` | Max tokens per completion |
| `LLM_WIKI_VAULT_PATH` | `vault` | Vault directory |
| `LLM_WIKI_DRY_RUN` | `false` | Global dry-run flag |
| `LLM_WIKI_VERBOSE` | `false` | Verbose output |

### RAG (Embedding + Reranking)

To enable RAG for precise document lookup (register addresses, spec values, etc.):

```bash
# .env
LLM_WIKI_EMBED_MODEL=qwen3-embedding          # or nomic-embed-text, text-embedding-ada-002, etc.
LLM_WIKI_EMBED_BASE_URL=http://localhost:11434/v1  # defaults to llm_base_url if empty
LLM_WIKI_RERANK_MODEL=qwen3-reranker          # optional — leave empty for cosine similarity only
LLM_WIKI_CHUNK_SIZE=800                       # chars per chunk (default)
LLM_WIKI_CHUNK_OVERLAP=150                    # overlap (default)
LLM_WIKI_RAG_TOP_K=5                          # chunks per query (default)
```

Once configured, `llm-wiki process` auto-embeds and `llm-wiki ask` uses both wiki + RAG:

```bash
llm-wiki add hardware-manual.pdf
llm-wiki process --latest     # normalize + ingest + embed (all in one)
llm-wiki ask "UART TX register address"
```

Or embed separately after ingestion:

```bash
llm-wiki embed --all          # embed all normalized sources
llm-wiki embed <source-id>    # embed a specific source
llm-wiki embed --all --force  # re-embed (e.g. after changing chunk_size)
```

#### How `ask` uses wiki + RAG

```
Question
  │
  ├─ Wiki keyword search → summary pages (concepts, entities, topics)
  │                        → structural understanding
  │
  └─ RAG embedding search → raw document chunks
      └─ optional rerank  → precise values (register addresses, specs)
                          → combined context → LLM answer
```

### Using a local LLM (Ollama)

```bash
LLM_WIKI_LLM_BASE_URL=http://localhost:11434/v1
LLM_WIKI_LLM_API_KEY=ollama
LLM_WIKI_LLM_MODEL=llama3.1
```

### Using LM Studio

```bash
LLM_WIKI_LLM_BASE_URL=http://localhost:1234/v1
LLM_WIKI_LLM_API_KEY=lm-studio
LLM_WIKI_LLM_MODEL=your-model-name
```

## Vault Structure

```
vault/
├── raw/                          # Immutable source files
│   ├── {source-id}.{ext}
│   └── {source-id}.meta.json     # Metadata sidecar
├── normalized/                   # Markitdown output
│   └── {source-id}.md
├── chunks/                       # RAG: chunked text (JSON)
│   └── {source-id}.json
├── embeddings/                   # RAG: embedding vectors (JSON)
│   └── {source-id}.json
├── wiki/
│   ├── index.md                  # Master table of contents
│   ├── log.md                    # Activity log
│   ├── overview.md               # Wiki summary
│   ├── sources/                  # Source summary pages
│   ├── entities/                 # People, orgs, products, places
│   ├── concepts/                 # Technical terms, methods
│   ├── topics/                   # Broad subject areas
│   ├── analyses/                 # Saved Q&A answers
│   └── reports/                  # Custom reports
└── schema/
    ├── AGENTS.md                 # LLM behavior instructions
    ├── wiki_schema.md            # Data schemas
    └── page_conventions.md      # Markdown formatting rules
```

## Sample Vault

The `sample_vault/` directory contains two mock source documents and the default schema files. To try them:

```bash
llm-wiki init
llm-wiki add sample_vault/raw/intro-to-transformers-mock.md --tag nlp --tag transformers
llm-wiki add sample_vault/raw/llm-scaling-laws-mock.txt --tag llm --tag scaling
llm-wiki normalize --all
llm-wiki ingest --all
llm-wiki ask "What are the key advantages of self-attention over recurrent layers?"
llm-wiki search "scaling laws"
llm-wiki status
llm-wiki lint
```

## Running Tests

```bash
pip install -e ".[dev]"
pytest
```

Tests use mocked LLM responses and temporary vaults — no API key required.

## Project Structure

```
llm_wiki/
├── cli.py              # Typer CLI app
├── config.py           # pydantic-settings configuration
├── llm.py              # LLM abstraction (OpenAI-compatible)
├── vault.py            # Vault operations
├── embedder.py         # Embedding client (OpenAI-compatible)
├── rag.py              # RAG index: chunking, vector search, reranking
├── schemas/
│   └── models.py       # Pydantic models for structured outputs
└── commands/
    ├── init_cmd.py
    ├── config_cmd.py
    ├── add_cmd.py
    ├── normalize_cmd.py
    ├── ingest_cmd.py
    ├── embed_cmd.py
    ├── ask_cmd.py
    ├── search_cmd.py
    ├── lint_cmd.py
    ├── status_cmd.py
    ├── log_cmd.py
    └── mcp_cmd.py
```

## Design Notes

- **Source IDs** are `{filename-slug}-{8-char-sha256}` — stable, unique, and human-readable
- **LLM calls use JSON mode** — structured outputs via `response_format: {type: json_object}`
- **Ingest is idempotent** — re-running updates pages rather than creating duplicates
- **Schema docs** are loaded into every LLM system prompt as behavioral constraints
- **Large documents are truncated** to 60K characters before LLM processing
- **Retry with backoff** on rate limits and transient server errors
