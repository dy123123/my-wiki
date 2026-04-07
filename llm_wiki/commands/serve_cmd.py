"""llm-wiki serve — local API server for Obsidian and other UIs."""

from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

from llm_wiki.config import Settings
from llm_wiki.vault import Vault

# ---------------------------------------------------------------------------
# Request / Response models (defined without FastAPI dependency at import time)
# ---------------------------------------------------------------------------

def run(vault: Vault, settings: Settings, host: str, port: int) -> None:
    try:
        import uvicorn
        from fastapi import FastAPI, HTTPException, Query
        from fastapi.middleware.cors import CORSMiddleware
        from pydantic import BaseModel
    except ImportError:
        from rich.console import Console
        Console().print(
            "[red]Error:[/red] serve requires extra deps.\n"
            "Install with: [bold]pip install 'llm-wiki[serve]'[/bold]"
        )
        raise SystemExit(1)

    from llm_wiki.llm import LLMClient

    llm = LLMClient(settings)

    app = FastAPI(
        title="llm-wiki API",
        description="Local API for the LLM-maintained markdown wiki",
        version="0.1.0",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],   # Obsidian runs as a local app — broad allow is fine
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # ------------------------------------------------------------------ #
    #  Models
    # ------------------------------------------------------------------ #

    class AskRequest(BaseModel):
        question: str
        save: bool = False

    class AskResponse(BaseModel):
        answer: str
        reasoning: str
        citations: list[str]
        confidence: str
        gaps: list[str]
        pages_consulted: int

    class SearchResult(BaseModel):
        path: str
        score: int
        snippet: str

    class SearchResponse(BaseModel):
        query: str
        results: list[SearchResult]

    class StatusResponse(BaseModel):
        vault_path: str
        sources_total: int
        sources_normalized: int
        sources_ingested: int
        wiki_pages: dict[str, int]
        vault_ok: bool

    class PageResponse(BaseModel):
        path: str
        content: str
        exists: bool

    # ------------------------------------------------------------------ #
    #  Routes
    # ------------------------------------------------------------------ #

    @app.get("/", tags=["meta"])
    def root():
        return {
            "name": "llm-wiki",
            "vault": str(vault.path),
            "model": settings.llm_model,
            "docs": f"http://{host}:{port}/docs",
        }

    @app.get("/status", response_model=StatusResponse, tags=["meta"])
    def status():
        if not vault.exists():
            raise HTTPException(status_code=503, detail="Vault not initialized")
        metas = vault.list_sources()
        wiki_counts: dict[str, int] = {}
        for subdir in [
            vault.wiki_sources, vault.wiki_entities, vault.wiki_concepts,
            vault.wiki_topics, vault.wiki_analyses, vault.wiki_reports,
        ]:
            wiki_counts[subdir.name] = len(list(subdir.glob("*.md"))) if subdir.exists() else 0
        return StatusResponse(
            vault_path=str(vault.path),
            sources_total=len(metas),
            sources_normalized=sum(1 for m in metas if vault.normalized_path(m.source_id).exists()),
            sources_ingested=sum(1 for m in metas if m.ingested_at),
            wiki_pages=wiki_counts,
            vault_ok=True,
        )

    @app.post("/ask", response_model=AskResponse, tags=["query"])
    def ask(req: AskRequest):
        if not vault.exists():
            raise HTTPException(status_code=503, detail="Vault not initialized")

        from llm_wiki.commands.ask_cmd import _find_relevant_pages, _ask_llm, _save_answer
        from llm_wiki.llm import LLMError

        relevant = _find_relevant_pages(req.question, vault)
        if not relevant:
            return AskResponse(
                answer="No relevant wiki pages found.",
                reasoning="The wiki does not contain pages matching your question. Try ingesting more sources.",
                citations=[],
                confidence="low",
                gaps=["No relevant content indexed yet"],
                pages_consulted=0,
            )

        MAX_PAGES = 8
        MAX_PAGE_CHARS = 6_000
        context_parts = []
        consulted_paths = []
        for page_path in relevant[:MAX_PAGES]:
            try:
                text = page_path.read_text(encoding="utf-8")[:MAX_PAGE_CHARS]
                rel = str(page_path.relative_to(vault.wiki))
                context_parts.append(f"### {rel}\n\n{text}")
                consulted_paths.append(rel)
            except Exception:
                pass

        context = "\n\n---\n\n".join(context_parts)

        try:
            result = _ask_llm(req.question, context, consulted_paths, vault.load_schema(), llm)
        except LLMError as e:
            raise HTTPException(status_code=502, detail=f"LLM error: {e}")

        if req.save:
            try:
                _save_answer(req.question, result, vault)
            except Exception:
                pass  # don't fail the response if save fails

        return AskResponse(
            answer=result.answer,
            reasoning=result.reasoning,
            citations=result.citations,
            confidence=result.confidence,
            gaps=result.gaps,
            pages_consulted=len(consulted_paths),
        )

    @app.get("/search", response_model=SearchResponse, tags=["query"])
    def search(q: str = Query(..., description="Search query"), n: int = Query(20, description="Max results")):
        if not vault.exists():
            raise HTTPException(status_code=503, detail="Vault not initialized")

        terms = re.findall(r"\w+", q.lower())
        if not terms:
            return SearchResponse(query=q, results=[])

        import re as _re
        pattern = _re.compile("|".join(_re.escape(t) for t in terms), _re.IGNORECASE)
        SNIPPET_WINDOW = 80
        scored: list[tuple[int, str, str]] = []

        for page_path in sorted(vault.wiki.rglob("*.md")):
            try:
                text = page_path.read_text(encoding="utf-8")
            except Exception:
                continue
            matches = list(pattern.finditer(text))
            if not matches:
                continue
            score = len(matches)
            m = matches[0]
            start = max(0, m.start() - SNIPPET_WINDOW)
            end = min(len(text), m.end() + SNIPPET_WINDOW)
            snippet = text[start:end].replace("\n", " ").strip()
            if start > 0:
                snippet = "…" + snippet
            if end < len(text):
                snippet += "…"
            rel = str(page_path.relative_to(vault.wiki))
            scored.append((score, rel, snippet))

        scored.sort(key=lambda x: x[0], reverse=True)
        return SearchResponse(
            query=q,
            results=[
                SearchResult(path=p, score=s, snippet=snip)
                for s, p, snip in scored[:n]
            ],
        )

    @app.get("/pages/{page_path:path}", response_model=PageResponse, tags=["content"])
    def get_page(page_path: str):
        target = (vault.wiki / page_path).resolve()
        # Security: ensure path stays inside vault/wiki
        try:
            target.relative_to(vault.wiki)
        except ValueError:
            raise HTTPException(status_code=400, detail="Path escapes vault")
        if not target.exists():
            return PageResponse(path=page_path, content="", exists=False)
        return PageResponse(
            path=page_path,
            content=target.read_text(encoding="utf-8"),
            exists=True,
        )

    @app.get("/index", tags=["content"])
    def get_index():
        if not vault.index.exists():
            raise HTTPException(status_code=404, detail="index.md not found")
        return {"content": vault.index.read_text(encoding="utf-8")}

    # ------------------------------------------------------------------ #
    #  Run
    # ------------------------------------------------------------------ #

    from rich.console import Console
    c = Console()
    c.print(f"[green]✓[/green] llm-wiki serve starting")
    c.print(f"  Vault : [cyan]{vault.path}[/cyan]")
    c.print(f"  Model : [cyan]{settings.llm_model}[/cyan]")
    c.print(f"  API   : [bold]http://{host}:{port}[/bold]")
    c.print(f"  Docs  : [bold]http://{host}:{port}/docs[/bold]")
    c.print()
    c.print("Obsidian Shell Commands example:")
    c.print(f'  [dim]curl -s -X POST http://{host}:{port}/ask -H "Content-Type: application/json" -d \'{{"question":"{{{{selection}}}}"}}\' | jq .answer[/dim]')
    c.print()

    uvicorn.run(app, host=host, port=port, log_level="warning")
