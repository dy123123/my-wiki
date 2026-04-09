"""llm-wiki web — combined web UI + REST API + MCP HTTP/SSE server."""

# NOTE: No `from __future__ import annotations` — FastAPI needs real types at decoration time.

import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from llm_wiki.config import Settings
from llm_wiki.vault import Vault

# ---------------------------------------------------------------------------
# In-memory task store
# ---------------------------------------------------------------------------
_tasks: dict[str, dict] = {}
_tasks_lock = threading.Lock()

# MCP connection tracking
_mcp_stats: dict = {"connections": 0, "last_connected": None, "tool_calls": []}
_mcp_lock = threading.Lock()


def _new_task(source_id: str, steps: list) -> str:
    task_id = str(uuid.uuid4())[:8]
    with _tasks_lock:
        _tasks[task_id] = {
            "task_id": task_id,
            "source_id": source_id,
            "status": "running",
            "current_step": steps[0] if steps else "",
            "message": "Starting…",
            "log": [],
        }
    return task_id


def _task_log(task_id: str, msg: str, status: str = "running", step: str = ""):
    with _tasks_lock:
        if task_id not in _tasks:
            return
        _tasks[task_id]["log"].append(msg)
        _tasks[task_id]["message"] = msg
        if status:
            _tasks[task_id]["status"] = status
        if step:
            _tasks[task_id]["current_step"] = step


def _track_mcp_call(tool: str, source: str):
    with _mcp_lock:
        _mcp_stats["tool_calls"] = (
            [{"tool": tool, "ts": _utcnow(), "source": source}] + _mcp_stats["tool_calls"]
        )[:20]


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run(
    settings: Settings,
    vault: Vault,
    host: str = "0.0.0.0",
    port: int = 7432,
    token: str = "",
) -> None:
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException, BackgroundTasks, Request, UploadFile, File
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.middleware.cors import CORSMiddleware
        from starlette.routing import Mount, Route
        from starlette.applications import Starlette
    except ImportError:
        from rich.console import Console
        Console().print(
            "[red]Error:[/red] web mode requires extra deps.\n"
            "Install with: [bold]pip install 'llm-wiki[web]'[/bold]"
        )
        raise SystemExit(1)

    # MCP setup
    try:
        from mcp.server import Server as McpServer
        from mcp import types as mcp_types
        from mcp.server.sse import SseServerTransport
        mcp_available = True
    except ImportError:
        mcp_available = False

    from llm_wiki.llm import LLMClient
    llm = LLMClient(settings)

    embedder = None
    rag_index = None
    if settings.embed_model:
        from llm_wiki.embedder import EmbedClient
        from llm_wiki.rag import RagIndex
        embedder = EmbedClient(settings)
        rag_index = RagIndex(vault)

    app = FastAPI(title="llm-wiki", docs_url="/api/docs")
    app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

    # ------------------------------------------------------------------ #
    # Auth middleware
    # ------------------------------------------------------------------ #

    @app.middleware("http")
    async def auth_middleware(request: Request, call_next):
        if not token:
            return await call_next(request)
        # Allow health check without token
        if request.url.path in ("/mcp/health", "/health"):
            return await call_next(request)
        auth = request.headers.get("Authorization", "")
        # Allow browser cookie session (simple)
        cookie = request.cookies.get("wiki_token", "")
        if auth == f"Bearer {token}" or cookie == token:
            return await call_next(request)
        # Redirect browser to login page
        if "text/html" in request.headers.get("accept", ""):
            return HTMLResponse(_login_html(port), status_code=200)
        return JSONResponse({"error": "Unauthorized"}, status_code=401)

    # ------------------------------------------------------------------ #
    # Web UI
    # ------------------------------------------------------------------ #

    @app.get("/", response_class=HTMLResponse)
    def index():
        return _ui_html(settings, port)

    @app.post("/api/login")
    async def login(request: Request):
        form = await request.form()
        t = form.get("token", "")
        if t == token:
            from fastapi.responses import RedirectResponse
            resp = RedirectResponse("/", status_code=302)
            resp.set_cookie("wiki_token", token, httponly=True, samesite="strict")
            return resp
        return HTMLResponse(_login_html(port, error=True), status_code=200)

    # ------------------------------------------------------------------ #
    # Sources API
    # ------------------------------------------------------------------ #

    @app.get("/api/sources")
    def list_sources():
        if not vault.exists():
            return {"sources": []}
        metas = vault.list_sources()
        ri = rag_index  # capture
        result = []
        for m in sorted(metas, key=lambda x: x.added_at, reverse=True):
            sid = m.source_id
            result.append({
                "source_id": sid,
                "title": m.title or m.original_name,
                "original_name": m.original_name,
                "extension": m.extension,
                "added_at": m.added_at,
                "normalized": vault.normalized_path(sid).exists(),
                "ingested": bool(m.ingested_at),
                "embedded": ri.is_indexed(sid) if ri else False,
                "tags": m.tags,
            })
        return {"sources": result}

    @app.post("/api/sources/upload")
    async def upload_source(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        auto_process: bool = True,
    ):
        if not vault.exists():
            raise HTTPException(400, "Vault not initialized. Run `llm-wiki init` first.")
        import tempfile, shutil
        suffix = Path(file.filename or "upload").suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            content = await file.read()
            tmp.write(content)
            tmp_path = Path(tmp.name)
        try:
            meta = vault.add_source(tmp_path, tags=[])
        finally:
            tmp_path.unlink(missing_ok=True)

        source_id = meta.source_id
        if auto_process:
            task_id = _new_task(source_id, ["normalize", "ingest", "embed"])
            background_tasks.add_task(_run_process, source_id, vault, settings, llm, embedder, rag_index, task_id)
            return {"source_id": source_id, "task_id": task_id, "status": "processing"}
        return {"source_id": source_id, "status": "added"}

    @app.delete("/api/sources/{source_id}")
    def delete_source(source_id: str):
        try:
            _delete_source(source_id, vault, rag_index)
        except FileNotFoundError as e:
            raise HTTPException(404, str(e))
        return {"status": "deleted", "source_id": source_id}

    @app.post("/api/sources/{source_id}/process")
    def process_source(source_id: str, background_tasks: BackgroundTasks):
        task_id = _new_task(source_id, ["normalize", "ingest", "embed"])
        background_tasks.add_task(_run_process, source_id, vault, settings, llm, embedder, rag_index, task_id)
        return {"task_id": task_id}

    @app.post("/api/sources/{source_id}/normalize")
    def normalize_source(source_id: str, background_tasks: BackgroundTasks):
        task_id = _new_task(source_id, ["normalize"])
        background_tasks.add_task(_run_step, "normalize", source_id, vault, settings, llm, embedder, rag_index, task_id)
        return {"task_id": task_id}

    @app.post("/api/sources/{source_id}/ingest")
    def ingest_source(source_id: str, background_tasks: BackgroundTasks):
        task_id = _new_task(source_id, ["ingest"])
        background_tasks.add_task(_run_step, "ingest", source_id, vault, settings, llm, embedder, rag_index, task_id)
        return {"task_id": task_id}

    @app.post("/api/sources/{source_id}/embed")
    def embed_source(source_id: str, background_tasks: BackgroundTasks):
        task_id = _new_task(source_id, ["embed"])
        background_tasks.add_task(_run_step, "embed", source_id, vault, settings, llm, embedder, rag_index, task_id)
        return {"task_id": task_id}

    @app.get("/api/tasks/{task_id}")
    def get_task(task_id: str):
        with _tasks_lock:
            t = _tasks.get(task_id)
        if not t:
            raise HTTPException(404, "Task not found")
        return t

    # ------------------------------------------------------------------ #
    # Wiki API
    # ------------------------------------------------------------------ #

    @app.get("/api/wiki")
    def list_wiki():
        if not vault.wiki.exists():
            return {"pages": []}
        pages = []
        for p in sorted(vault.wiki.rglob("*.md")):
            rel = str(p.relative_to(vault.wiki))
            pages.append({"path": rel, "size": p.stat().st_size})
        return {"pages": pages}

    @app.get("/api/wiki/{page_path:path}")
    def get_wiki_page(page_path: str):
        target = (vault.wiki / page_path).resolve()
        try:
            target.relative_to(vault.wiki)
        except ValueError:
            raise HTTPException(400, "Path escapes vault")
        if not target.exists():
            raise HTTPException(404, "Page not found")
        return {"path": page_path, "content": target.read_text(encoding="utf-8")}

    # ------------------------------------------------------------------ #
    # Ask API
    # ------------------------------------------------------------------ #

    @app.post("/api/ask")
    async def ask_question(request: Request):
        body = await request.json()
        question = body.get("question", "").strip()
        if not question:
            raise HTTPException(400, "question required")

        import asyncio
        result = await asyncio.to_thread(_ask, question, vault, llm, embedder, rag_index)
        _track_mcp_call("wiki_ask", "web")
        return result

    # ------------------------------------------------------------------ #
    # MCP status API
    # ------------------------------------------------------------------ #

    @app.get("/api/mcp/status")
    def mcp_status():
        with _mcp_lock:
            return dict(_mcp_stats, available=mcp_available)

    @app.get("/health")
    @app.get("/mcp/health")
    def health():
        return {"status": "ok", "vault": str(vault.path)}

    # ------------------------------------------------------------------ #
    # MCP SSE (mounted on /mcp)
    # ------------------------------------------------------------------ #

    if mcp_available:
        mcp_server = _build_mcp_server(vault, llm, embedder, rag_index, mcp_types)
        sse_transport = SseServerTransport("/mcp/messages/")

        @app.get("/mcp/sse")
        async def mcp_sse_endpoint(request: Request):
            if token:
                auth = request.headers.get("Authorization", "")
                if auth != f"Bearer {token}":
                    return JSONResponse({"error": "Unauthorized"}, status_code=401)
            with _mcp_lock:
                _mcp_stats["connections"] += 1
                _mcp_stats["last_connected"] = _utcnow()
            try:
                async with sse_transport.connect_sse(
                    request.scope, request.receive, request._send
                ) as streams:
                    await mcp_server.run(streams[0], streams[1], mcp_server.create_initialization_options())
            finally:
                with _mcp_lock:
                    _mcp_stats["connections"] = max(0, _mcp_stats["connections"] - 1)

        app.mount("/mcp/messages", sse_transport.handle_post_message)

    # ------------------------------------------------------------------ #
    # Start
    # ------------------------------------------------------------------ #

    from rich.console import Console
    c = Console()
    c.print(f"[green]✓[/green] llm-wiki web starting")
    c.print(f"  UI    : [bold]http://{host}:{port}[/bold]")
    c.print(f"  API   : http://{host}:{port}/api/docs")
    if mcp_available:
        c.print(f"  MCP   : http://{host}:{port}/mcp/sse")
    if token:
        c.print(f"  Auth  : Bearer token enabled")
    c.print()

    uvicorn.run(app, host=host, port=port, log_level="warning")


# ---------------------------------------------------------------------------
# Background pipeline runners
# ---------------------------------------------------------------------------

def _run_process(source_id, vault, settings, llm, embedder, rag_index, task_id):
    for step in ["normalize", "ingest", "embed"]:
        if step == "embed" and not (embedder and embedder.enabled):
            continue
        _run_step(step, source_id, vault, settings, llm, embedder, rag_index, task_id)
        with _tasks_lock:
            if _tasks.get(task_id, {}).get("status") == "error":
                return
    _task_log(task_id, "Done.", "done")


def _run_step(step, source_id, vault, settings, llm, embedder, rag_index, task_id):
    import io, contextlib
    _task_log(task_id, f"Running {step}…", step=step)
    try:
        if step == "normalize":
            from llm_wiki.commands.normalize_cmd import run as _run
            buf = io.StringIO()
            with contextlib.redirect_stdout(buf):
                _run(source_id, vault, False, False)
            _task_log(task_id, f"Normalize done.", step=step)

        elif step == "ingest":
            from llm_wiki.commands.ingest_cmd import run as _run
            _run(source_id, vault, llm, False, False, False)
            _task_log(task_id, f"Ingest done.", step=step)

        elif step == "embed":
            if not (embedder and embedder.enabled):
                _task_log(task_id, "Embed skipped (no embed model).", step=step)
                return
            from llm_wiki.commands.embed_cmd import run as _run
            from llm_wiki.rag import RagIndex
            ri = rag_index or RagIndex(vault)
            _run(source_id, vault, embedder, False, True, False)
            _task_log(task_id, f"Embed done.", step=step)

    except Exception as e:
        _task_log(task_id, f"{step} error: {e}", "error")


# ---------------------------------------------------------------------------
# Source deletion
# ---------------------------------------------------------------------------

def _delete_source(source_id: str, vault: Vault, rag_index) -> None:
    """Delete a source and all derived files."""
    import glob as _glob

    # Verify exists
    meta = vault.load_meta(source_id)  # raises FileNotFoundError if missing

    # Raw file
    for f in vault.raw.glob(f"{source_id}.*"):
        if not f.name.endswith(".meta.json"):
            f.unlink(missing_ok=True)
    # Meta
    meta_path = vault.raw / f"{source_id}.meta.json"
    meta_path.unlink(missing_ok=True)

    # Normalized
    norm = vault.normalized_path(source_id)
    norm.unlink(missing_ok=True)

    # Wiki source page
    vault.source_page_path(source_id).unlink(missing_ok=True)

    # Chunks + embeddings
    if rag_index:
        rag_index.chunk_path(source_id).unlink(missing_ok=True)
        rag_index.embed_path(source_id).unlink(missing_ok=True)
    else:
        (vault.chunks / f"{source_id}.json").unlink(missing_ok=True)
        (vault.embeddings / f"{source_id}.npy").unlink(missing_ok=True)

    # Log
    vault.append_log(f"- Deleted source **{source_id}** ({meta.original_name})")


# ---------------------------------------------------------------------------
# Ask helper
# ---------------------------------------------------------------------------

def _ask(question: str, vault: Vault, llm, embedder, rag_index) -> dict:
    from llm_wiki.commands.ask_cmd import _find_relevant_pages, _ask_llm
    from llm_wiki.llm import LLMError

    relevant = _find_relevant_pages(question, vault)
    rag_chunks = []
    if embedder and getattr(embedder, "enabled", False) and rag_index:
        try:
            q_emb = embedder.embed([question])
            if q_emb:
                candidates = rag_index.search(q_emb[0], query_text=question, top_k=40)
                rag_chunks = rag_index.rerank_and_trim(question, candidates, embedder, top_k=5)
        except Exception:
            pass

    context_parts = []
    consulted = []
    for p in relevant[:6]:
        try:
            text = p.read_text(encoding="utf-8")[:4000]
            rel = str(p.relative_to(vault.wiki))
            context_parts.append(f"### [Wiki] {rel}\n\n{text}")
            consulted.append(rel)
        except Exception:
            pass
    if rag_chunks:
        context_parts.append("### [Source Chunks]")
        for chunk in rag_chunks:
            context_parts.append(f"**{chunk.source_id}** chunk {chunk.chunk_idx}\n\n{chunk.text[:1200]}")

    if not context_parts:
        return {"answer": "No relevant content found.", "confidence": "low", "citations": [], "gaps": []}

    try:
        result = _ask_llm(question, "\n\n---\n\n".join(context_parts), consulted, vault.load_schema(), llm)
        return {
            "answer": result.answer,
            "reasoning": result.reasoning,
            "citations": result.citations,
            "confidence": result.confidence,
            "gaps": result.gaps,
            "rag_chunks": len(rag_chunks),
        }
    except LLMError as e:
        return {"answer": f"LLM error: {e}", "confidence": "low", "citations": [], "gaps": []}


# ---------------------------------------------------------------------------
# MCP server builder (shared tools with mcp_cmd.py)
# ---------------------------------------------------------------------------

def _build_mcp_server(vault, llm, embedder, rag_index, mcp_types):
    from mcp.server import Server

    server = Server("llm-wiki")

    @server.list_tools()
    async def list_tools():
        return [
            mcp_types.Tool(
                name="wiki_ask",
                description="Ask a question against the local wiki + RAG knowledge base.",
                inputSchema={"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]},
            ),
            mcp_types.Tool(
                name="wiki_search",
                description="Full-text search across wiki pages.",
                inputSchema={"type": "object", "properties": {"query": {"type": "string"}, "max_results": {"type": "integer", "default": 10}}, "required": ["query"]},
            ),
            mcp_types.Tool(
                name="wiki_page",
                description="Read a wiki page by path (relative to wiki/).",
                inputSchema={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
            ),
            mcp_types.Tool(
                name="wiki_status",
                description="Show vault status.",
                inputSchema={"type": "object", "properties": {}},
            ),
        ]

    @server.call_tool()
    async def call_tool(name: str, arguments: dict):
        import asyncio, re
        try:
            if name == "wiki_ask":
                q = arguments["question"]
                _track_mcp_call("wiki_ask", "mcp")
                result = await asyncio.to_thread(_ask, q, vault, llm, embedder, rag_index)
                text = f"**{result['answer']}**\n\n{result.get('reasoning','')}"
                if result.get("citations"):
                    text += "\n\nSources: " + ", ".join(f"`{c}`" for c in result["citations"])
            elif name == "wiki_search":
                _track_mcp_call("wiki_search", "mcp")
                query = arguments["query"]
                terms = re.findall(r"\w+", query.lower())
                pattern = re.compile("|".join(re.escape(t) for t in terms), re.IGNORECASE)
                scored = []
                for p in vault.wiki.rglob("*.md"):
                    try:
                        content = p.read_text(encoding="utf-8")
                        ms = list(pattern.finditer(content))
                        if ms:
                            m = ms[0]
                            snip = content[max(0, m.start()-80):m.end()+80].replace("\n", " ")
                            scored.append((len(ms), str(p.relative_to(vault.wiki)), snip))
                    except Exception:
                        pass
                scored.sort(reverse=True)
                n = arguments.get("max_results", 10)
                text = f"Results for '{query}':\n" + "\n".join(f"**{p}** ({s})\n…{snip}…" for s, p, snip in scored[:n])
            elif name == "wiki_page":
                _track_mcp_call("wiki_page", "mcp")
                target = (vault.wiki / arguments["path"]).resolve()
                target.relative_to(vault.wiki)
                text = target.read_text(encoding="utf-8") if target.exists() else "Page not found."
            elif name == "wiki_status":
                _track_mcp_call("wiki_status", "mcp")
                metas = vault.list_sources()
                ri = rag_index
                text = (
                    f"Vault: {vault.path}\n"
                    f"Sources: {len(metas)} total | "
                    f"{sum(1 for m in metas if vault.normalized_path(m.source_id).exists())} normalized | "
                    f"{sum(1 for m in metas if m.ingested_at)} ingested | "
                    f"{len(ri.indexed_sources()) if ri else 0} embedded"
                )
            else:
                text = f"Unknown tool: {name}"
        except Exception as e:
            text = f"Error: {e}"
        return [mcp_types.TextContent(type="text", text=text)]

    return server


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------

def _login_html(port: int, error: bool = False) -> str:
    err = '<p class="text-red-400 text-sm mt-2">Invalid token.</p>' if error else ""
    return f"""<!DOCTYPE html><html><head><title>llm-wiki login</title>
<meta charset="UTF-8"><script src="https://cdn.tailwindcss.com"></script></head>
<body class="bg-gray-950 text-gray-100 min-h-screen flex items-center justify-center font-mono">
<div class="bg-gray-900 border border-gray-700 rounded-lg p-8 w-80">
  <h1 class="text-blue-400 font-bold text-xl mb-6">llm-wiki</h1>
  <form method="POST" action="/api/login">
    <input name="token" type="password" placeholder="Access token"
      class="w-full bg-gray-800 border border-gray-600 rounded px-3 py-2 text-sm mb-3 focus:border-blue-500 outline-none">
    {err}
    <button class="w-full bg-blue-600 hover:bg-blue-700 py-2 rounded text-sm mt-2">Login</button>
  </form>
</div></body></html>"""


def _ui_html(settings: Settings, port: int) -> str:
    return """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>llm-wiki</title>
<script src="https://cdn.tailwindcss.com"></script>
<style>
  .tab-active { border-bottom: 2px solid #3b82f6; color: #60a5fa; }
  .badge-ok { background:#16a34a; }
  .badge-no { background:#374151; color:#6b7280; }
  .badge-run { background:#d97706; animation: pulse 1s infinite; }
  @keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.5} }
  pre { white-space: pre-wrap; word-break: break-word; }
</style>
</head>
<body class="bg-gray-950 text-gray-100 min-h-screen font-mono text-sm">

<!-- Header -->
<header class="border-b border-gray-800 px-6 py-3 flex items-center justify-between">
  <div class="flex items-center gap-3">
    <span class="text-blue-400 font-bold text-base">llm-wiki</span>
    <span class="text-gray-500 text-xs" id="vault-path"></span>
  </div>
  <div class="flex items-center gap-2 text-xs">
    <span class="w-2 h-2 rounded-full" id="mcp-dot" style="background:#374151"></span>
    <span class="text-gray-400" id="mcp-label">MCP</span>
  </div>
</header>

<!-- Tabs -->
<div class="border-b border-gray-800 px-4 flex">
  <button onclick="showTab('sources')" id="tab-sources" class="px-4 py-2 text-gray-400 hover:text-gray-200 tab-active">Sources</button>
  <button onclick="showTab('wiki')" id="tab-wiki" class="px-4 py-2 text-gray-400 hover:text-gray-200">Wiki</button>
  <button onclick="showTab('ask')" id="tab-ask" class="px-4 py-2 text-gray-400 hover:text-gray-200">Ask</button>
</div>

<!-- Sources -->
<div id="pane-sources" class="p-5">
  <div id="upload-zone"
    class="border-2 border-dashed border-gray-700 rounded-lg p-6 text-center mb-5 cursor-pointer hover:border-gray-500 transition-colors"
    onclick="document.getElementById('file-input').click()"
    ondragover="ev.preventDefault();this.classList.add('border-blue-500')"
    ondragleave="this.classList.remove('border-blue-500')"
    ondrop="handleDrop(event)">
    <p class="text-gray-400">Drag &amp; drop files or <span class="text-blue-400">click to upload</span></p>
    <p class="text-gray-600 text-xs mt-1">PDF, DOCX, MD, TXT, and more — will auto-process</p>
    <input type="file" id="file-input" class="hidden" multiple onchange="uploadFiles(this.files)">
  </div>
  <div id="source-list"><p class="text-gray-500">Loading…</p></div>
</div>

<!-- Wiki -->
<div id="pane-wiki" class="p-5 hidden">
  <div class="flex gap-4">
    <div class="w-56 flex-shrink-0 overflow-y-auto max-h-screen">
      <div id="wiki-tree" class="text-xs text-gray-400"></div>
    </div>
    <div class="flex-1">
      <p class="text-gray-500 text-xs mb-2" id="wiki-page-path"></p>
      <pre id="wiki-content" class="text-xs text-gray-300 bg-gray-900 rounded p-4 max-h-screen overflow-y-auto"></pre>
    </div>
  </div>
</div>

<!-- Ask -->
<div id="pane-ask" class="p-5 hidden max-w-3xl">
  <div class="flex gap-2 mb-4">
    <input id="ask-input" type="text" placeholder="Ask a question…"
      class="flex-1 bg-gray-900 border border-gray-700 rounded px-3 py-2 focus:border-blue-500 outline-none"
      onkeydown="if(event.key==='Enter')doAsk()">
    <button onclick="doAsk()" class="bg-blue-600 hover:bg-blue-700 px-4 py-2 rounded">Ask</button>
  </div>
  <div id="ask-result"></div>
</div>

<script>
const API = '';
let pollTimers = {};

// ── Tabs ──────────────────────────────────────────────────────────────────
function showTab(name) {
  ['sources','wiki','ask'].forEach(t => {
    document.getElementById('pane-'+t).classList.toggle('hidden', t !== name);
    document.getElementById('tab-'+t).classList.toggle('tab-active', t === name);
  });
  if (name === 'sources') loadSources();
  if (name === 'wiki') loadWikiTree();
}

// ── Sources ───────────────────────────────────────────────────────────────
async function loadSources() {
  const r = await fetch(API+'/api/sources');
  const {sources} = await r.json();
  const el = document.getElementById('source-list');
  if (!sources.length) { el.innerHTML = '<p class="text-gray-500">No sources yet. Upload a file above.</p>'; return; }
  el.innerHTML = sources.map(s => sourceRow(s)).join('');
}

function sourceRow(s) {
  const b = (ok, label) => `<span class="px-1.5 py-0.5 rounded text-xs font-bold ${ok ? 'badge-ok' : 'badge-no'}">${label}</span>`;
  return `<div class="flex items-center gap-3 py-2 border-b border-gray-800" id="row-${s.source_id}">
    <div class="flex-1 min-w-0">
      <p class="truncate text-gray-200">${s.title || s.original_name}</p>
      <p class="text-gray-500 text-xs">${s.source_id} · ${s.added_at.slice(0,10)}</p>
    </div>
    <div class="flex gap-1 items-center flex-shrink-0">
      ${b(s.normalized,'N')} ${b(s.ingested,'I')} ${b(s.embedded,'E')}
    </div>
    <div class="flex gap-1 flex-shrink-0">
      <button onclick="runStep('${s.source_id}','process')" class="px-2 py-1 bg-blue-700 hover:bg-blue-600 rounded text-xs">Process</button>
      <button onclick="runStep('${s.source_id}','normalize')" class="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs">N</button>
      <button onclick="runStep('${s.source_id}','ingest')" class="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs">I</button>
      <button onclick="runStep('${s.source_id}','embed')" class="px-2 py-1 bg-gray-700 hover:bg-gray-600 rounded text-xs">E</button>
      <button onclick="deleteSource('${s.source_id}')" class="px-2 py-1 bg-red-900 hover:bg-red-700 rounded text-xs">✕</button>
    </div>
  </div>`;
}

async function runStep(sourceId, step) {
  const url = step === 'process'
    ? `/api/sources/${sourceId}/process`
    : `/api/sources/${sourceId}/${step}`;
  const r = await fetch(API+url, {method:'POST'});
  const {task_id} = await r.json();
  pollTask(task_id, sourceId);
}

function pollTask(taskId, sourceId) {
  if (pollTimers[taskId]) clearInterval(pollTimers[taskId]);
  setRowStatus(sourceId, 'running');
  pollTimers[taskId] = setInterval(async () => {
    const r = await fetch(API+`/api/tasks/${taskId}`);
    const t = await r.json();
    setRowMsg(sourceId, t.message);
    if (t.status === 'done' || t.status === 'error') {
      clearInterval(pollTimers[taskId]);
      delete pollTimers[taskId];
      setRowMsg(sourceId, '');
      await loadSources();
    }
  }, 1500);
}

function setRowStatus(sourceId, status) {
  const row = document.getElementById('row-'+sourceId);
  if (row) row.style.opacity = status === 'running' ? '0.7' : '1';
}

function setRowMsg(sourceId, msg) {
  const row = document.getElementById('row-'+sourceId);
  if (!row) return;
  let el = row.querySelector('.task-msg');
  if (!el) { el = document.createElement('span'); el.className = 'task-msg text-yellow-400 text-xs'; row.prepend(el); }
  el.textContent = msg;
}

async function deleteSource(sourceId) {
  if (!confirm(`Delete ${sourceId}?`)) return;
  await fetch(API+`/api/sources/${sourceId}`, {method:'DELETE'});
  await loadSources();
}

async function uploadFiles(files) {
  for (const file of files) {
    const fd = new FormData();
    fd.append('file', file);
    const r = await fetch(API+'/api/sources/upload', {method:'POST', body:fd});
    const data = await r.json();
    await loadSources();
    if (data.task_id) pollTask(data.task_id, data.source_id);
  }
}

function handleDrop(ev) {
  ev.preventDefault();
  document.getElementById('upload-zone').classList.remove('border-blue-500');
  uploadFiles(ev.dataTransfer.files);
}

// ── Wiki ──────────────────────────────────────────────────────────────────
async function loadWikiTree() {
  const r = await fetch(API+'/api/wiki');
  const {pages} = await r.json();
  const tree = document.getElementById('wiki-tree');
  tree.innerHTML = pages.map(p =>
    `<div class="py-0.5 cursor-pointer hover:text-blue-400 truncate" onclick="loadWikiPage('${p.path}')">${p.path}</div>`
  ).join('');
}

async function loadWikiPage(path) {
  document.getElementById('wiki-page-path').textContent = path;
  const r = await fetch(API+`/api/wiki/${path}`);
  const {content} = await r.json();
  document.getElementById('wiki-content').textContent = content;
}

// ── Ask ───────────────────────────────────────────────────────────────────
async function doAsk() {
  const q = document.getElementById('ask-input').value.trim();
  if (!q) return;
  const el = document.getElementById('ask-result');
  el.innerHTML = '<p class="text-yellow-400">Thinking…</p>';
  const r = await fetch(API+'/api/ask', {
    method:'POST', headers:{'Content-Type':'application/json'},
    body: JSON.stringify({question:q})
  });
  const d = await r.json();
  el.innerHTML = `
    <div class="bg-gray-900 rounded p-4 mb-3">
      <p class="font-bold text-white mb-2">${d.answer}</p>
      <p class="text-gray-300">${d.reasoning || ''}</p>
    </div>
    ${d.citations?.length ? '<p class="text-xs text-gray-500">Sources: ' + d.citations.map(c=>`<code>${c}</code>`).join(', ') + '</p>' : ''}
    <span class="text-xs text-gray-600">Confidence: ${d.confidence} · RAG chunks: ${d.rag_chunks ?? 0}</span>
  `;
}

// ── MCP status ────────────────────────────────────────────────────────────
async function refreshMcpStatus() {
  try {
    const r = await fetch(API+'/api/mcp/status');
    const d = await r.json();
    const dot = document.getElementById('mcp-dot');
    const label = document.getElementById('mcp-label');
    if (!d.available) { dot.style.background='#6b7280'; label.textContent='MCP unavailable'; return; }
    const on = d.connections > 0;
    dot.style.background = on ? '#16a34a' : '#6b7280';
    label.textContent = `MCP ${on ? d.connections+' connected' : 'idle'}`;
  } catch(e) {}
}

// ── Init ──────────────────────────────────────────────────────────────────
(async () => {
  try {
    const r = await fetch(API+'/health');
    const d = await r.json();
    document.getElementById('vault-path').textContent = d.vault || '';
  } catch(e) {}
  loadSources();
  refreshMcpStatus();
  setInterval(refreshMcpStatus, 5000);
})();
</script>
</body>
</html>"""
