"""llm-wiki mcp — MCP server (stdio or HTTP/SSE) for OpenCode, Claude Desktop, Cursor, etc."""

from __future__ import annotations

import asyncio
import re
from typing import Optional

from llm_wiki.config import Settings
from llm_wiki.vault import Vault


def run(
    settings: Settings,
    vault: Vault,
    http: bool = False,
    host: str = "0.0.0.0",
    port: int = 8080,
    token: str = "",
) -> None:
    try:
        from mcp.server import Server
        from mcp import types as mcp_types
        if not http:
            from mcp.server.stdio import stdio_server
    except ImportError:
        from rich.console import Console
        Console().print(
            "[red]Error:[/red] MCP not installed.\n"
            "Install with: [bold]pip install 'llm-wiki[mcp]'[/bold]"
        )
        raise SystemExit(1)

    if http:
        try:
            import uvicorn
            from starlette.applications import Starlette
            from starlette.middleware import Middleware
            from starlette.middleware.base import BaseHTTPMiddleware
            from starlette.requests import Request
            from starlette.responses import JSONResponse
            from starlette.routing import Mount, Route
            from mcp.server.sse import SseServerTransport
        except ImportError:
            from rich.console import Console
            Console().print(
                "[red]Error:[/red] HTTP mode requires extra deps.\n"
                "Install with: [bold]pip install 'llm-wiki[mcp]'[/bold]"
            )
            raise SystemExit(1)

    from llm_wiki.llm import LLMClient
    llm = LLMClient(settings)

    embedder = None
    rag_index = None
    if settings.embed_model:
        from llm_wiki.embedder import EmbedClient
        from llm_wiki.rag import RagIndex
        embedder = EmbedClient(settings)
        rag_index = RagIndex(vault)

    server = Server("llm-wiki")

    @server.list_tools()
    async def list_tools() -> list[mcp_types.Tool]:
        return [
            mcp_types.Tool(
                name="wiki_ask",
                description=(
                    "Ask a question against the local wiki knowledge base. "
                    "Searches wiki summary pages first, then raw document chunks (RAG) if configured. "
                    "Use for: technical questions, concept explanations, spec lookups, register addresses."
                ),
                inputSchema={
                    "type": "object",
                    "properties": {
                        "question": {"type": "string", "description": "The question to answer"},
                    },
                    "required": ["question"],
                },
            ),
            mcp_types.Tool(
                name="wiki_search",
                description="Full-text keyword search across wiki pages. Returns matching paths and snippets.",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "query": {"type": "string", "description": "Search query"},
                        "max_results": {"type": "integer", "default": 10},
                    },
                    "required": ["query"],
                },
            ),
            mcp_types.Tool(
                name="wiki_page",
                description="Read a specific wiki page. Path is relative to wiki/ (e.g. 'concepts/transformer.md').",
                inputSchema={
                    "type": "object",
                    "properties": {
                        "path": {"type": "string", "description": "Page path relative to wiki/"},
                    },
                    "required": ["path"],
                },
            ),
            mcp_types.Tool(
                name="wiki_status",
                description="Show vault status: number of sources, wiki pages, and pipeline progress.",
                inputSchema={
                    "type": "object",
                    "properties": {},
                },
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict) -> list[mcp_types.TextContent]:
        try:
            if name == "wiki_ask":
                text = await asyncio.to_thread(
                    _wiki_ask, arguments["question"], vault, llm, embedder, rag_index
                )
            elif name == "wiki_search":
                text = await asyncio.to_thread(
                    _wiki_search, arguments["query"], vault, arguments.get("max_results", 10)
                )
            elif name == "wiki_page":
                text = await asyncio.to_thread(_wiki_page, arguments["path"], vault)
            elif name == "wiki_status":
                text = await asyncio.to_thread(_wiki_status, vault, rag_index)
            else:
                text = f"Unknown tool: {name}"
        except Exception as e:
            text = f"Error: {e}"

        return [mcp_types.TextContent(type="text", text=text)]

    if http:
        _run_http(server, mcp_types, host, port, token, uvicorn, Starlette, Middleware,
                  BaseHTTPMiddleware, Request, JSONResponse, Route, Mount, SseServerTransport)
    else:
        asyncio.run(_serve_stdio(server, stdio_server))


async def _serve_stdio(server, stdio_server) -> None:
    async with stdio_server() as (read_stream, write_stream):
        await server.run(read_stream, write_stream, server.create_initialization_options())


def _run_http(server, mcp_types, host, port, token, uvicorn, Starlette, Middleware,
              BaseHTTPMiddleware, Request, JSONResponse, Route, Mount, SseServerTransport) -> None:
    from rich.console import Console
    c = Console()

    init_options = server.create_initialization_options()
    sse_transport = SseServerTransport("/messages/")

    async def handle_sse(request: Request):
        # Token auth
        if token:
            auth = request.headers.get("Authorization", "")
            if auth != f"Bearer {token}":
                return JSONResponse({"error": "Unauthorized"}, status_code=401)
        async with sse_transport.connect_sse(
            request.scope, request.receive, request._send
        ) as streams:
            await server.run(streams[0], streams[1], init_options)

    class AuthMiddleware(BaseHTTPMiddleware):
        async def dispatch(self, request: Request, call_next):
            # Only protect /messages/ POST (SSE is handled above)
            if token and request.url.path.startswith("/messages/"):
                auth = request.headers.get("Authorization", "")
                if auth != f"Bearer {token}":
                    return JSONResponse({"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

    middleware = [Middleware(AuthMiddleware)] if token else []

    app = Starlette(
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse_transport.handle_post_message),
            Route("/health", endpoint=lambda r: JSONResponse({"status": "ok"})),
        ],
        middleware=middleware,
    )

    c.print(f"[green]✓[/green] llm-wiki MCP HTTP server starting")
    c.print(f"  SSE endpoint : [bold]http://{host}:{port}/sse[/bold]")
    c.print(f"  Health check : http://{host}:{port}/health")
    if token:
        c.print(f"  Auth         : Bearer token enabled")
    c.print()
    c.print("[dim]OpenCode config:[/dim]")
    auth_line = f'\n      "headers": {{"Authorization": "Bearer {token}"}}' if token else ""
    c.print(f'[dim]  {{"mcpServers": {{"my-wiki": {{"type":"sse","url":"http://<server>:{port}/sse"{auth_line}}}}}}}[/dim]')
    c.print()

    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Tool implementations (synchronous — run in thread via asyncio.to_thread)
# ---------------------------------------------------------------------------

def _wiki_ask(
    question: str,
    vault: Vault,
    llm,
    embedder: Optional[object],
    rag_index: Optional[object],
) -> str:
    from llm_wiki.commands.ask_cmd import _find_relevant_pages, _ask_llm
    from llm_wiki.llm import LLMError

    relevant = _find_relevant_pages(question, vault)

    rag_chunks = []
    if embedder is not None and getattr(embedder, "enabled", False) and rag_index is not None:
        try:
            q_emb = embedder.embed([question])
            if q_emb:
                candidates = rag_index.search(q_emb[0], query_text=question, top_k=40)
                rag_chunks = rag_index.rerank_and_trim(question, candidates, embedder, top_k=5)
        except Exception:
            pass

    if not relevant and not rag_chunks:
        return (
            "No relevant content found in the wiki. "
            "Run `llm-wiki process --all` to ingest documents first."
        )

    MAX_PAGES = 6
    MAX_PAGE_CHARS = 4_000
    MAX_CHUNK_CHARS = 1_200
    context_parts: list[str] = []
    consulted: list[str] = []

    for page_path in relevant[:MAX_PAGES]:
        try:
            text = page_path.read_text(encoding="utf-8")[:MAX_PAGE_CHARS]
            rel = str(page_path.relative_to(vault.wiki))
            context_parts.append(f"### [Wiki] {rel}\n\n{text}")
            consulted.append(rel)
        except Exception:
            pass

    if rag_chunks:
        context_parts.append("### [Source Chunks — raw document excerpts]")
        for chunk in rag_chunks:
            context_parts.append(
                f"**{chunk.source_id}** chunk {chunk.chunk_idx} (score {chunk.score:.3f})\n\n"
                + chunk.text[:MAX_CHUNK_CHARS]
            )

    context = "\n\n---\n\n".join(context_parts)

    try:
        result = _ask_llm(question, context, consulted, vault.load_schema(), llm)
    except LLMError as e:
        return f"LLM error: {e}"

    out = f"**{result.answer}**\n\n{result.reasoning}"
    if result.citations:
        out += "\n\n**Sources:** " + ", ".join(f"`{c}`" for c in result.citations)
    out += f"\n\n**Confidence:** {result.confidence}"
    if result.gaps:
        out += "\n\n**Gaps:** " + "; ".join(result.gaps)
    return out


def _wiki_search(query: str, vault: Vault, max_results: int) -> str:
    terms = re.findall(r"\w+", query.lower())
    if not terms:
        return "No search terms provided."

    pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
    SNIPPET = 100
    scored: list[tuple[int, str, str]] = []

    for page_path in vault.wiki.rglob("*.md"):
        try:
            text = page_path.read_text(encoding="utf-8")
        except Exception:
            continue
        matches = list(pattern.finditer(text))
        if not matches:
            continue
        m = matches[0]
        start = max(0, m.start() - SNIPPET)
        end = min(len(text), m.end() + SNIPPET)
        snippet = text[start:end].replace("\n", " ").strip()
        rel = str(page_path.relative_to(vault.wiki))
        scored.append((len(matches), rel, snippet))

    scored.sort(reverse=True)
    if not scored:
        return f"No results found for: {query}"

    lines = [f"Search results for '{query}' ({min(len(scored), max_results)} of {len(scored)}):"]
    for score, path, snippet in scored[:max_results]:
        lines.append(f"\n**{path}** ({score} match{'es' if score > 1 else ''})\n…{snippet}…")
    return "\n".join(lines)


def _wiki_page(path: str, vault: Vault) -> str:
    target = (vault.wiki / path).resolve()
    try:
        target.relative_to(vault.wiki)
    except ValueError:
        return "Error: path escapes vault boundary"
    if not target.exists():
        return f"Page not found: {path}"
    return target.read_text(encoding="utf-8")


def _wiki_status(vault: Vault, rag_index: Optional[object]) -> str:
    if not vault.exists():
        return "Vault not initialized. Run `llm-wiki init` first."

    metas = vault.list_sources()
    normalized = sum(1 for m in metas if vault.normalized_path(m.source_id).exists())
    ingested = sum(1 for m in metas if m.ingested_at)
    embedded = len(rag_index.indexed_sources()) if rag_index is not None else 0

    wiki_counts: dict[str, int] = {}
    for subdir in [vault.wiki_sources, vault.wiki_entities, vault.wiki_concepts, vault.wiki_topics]:
        wiki_counts[subdir.name] = len(list(subdir.glob("*.md"))) if subdir.exists() else 0

    lines = [
        f"**Vault:** {vault.path}",
        f"**Sources:** {len(metas)} total | {normalized} normalized | {ingested} ingested | {embedded} embedded (RAG)",
        "**Wiki pages:** " + " | ".join(f"{k}: {v}" for k, v in wiki_counts.items()),
    ]
    return "\n".join(lines)
