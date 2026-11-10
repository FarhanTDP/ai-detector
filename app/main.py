"""FastAPI application: JSON API + static frontend.

Run with:
    python -m app                      (reads AIDET_HOST / AIDET_PORT)
or
    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging
import secrets
import socket
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles

from . import __version__
from .config import Settings
from .database import Database, VersionConflict
from .detector import BaseDetector, HFDetector, ModelNotReady, TextTooLong, TextTooShort
from .schemas import DocumentCreate, DocumentUpdate, ScanRequest

log = logging.getLogger("scribescan")
TOKEN_COOKIE = "scribescan_token"

LOCKED_PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>ScribeScan — locked</title>
<style>body{background:#0f1115;color:#e6e8ee;font:16px system-ui,sans-serif;display:grid;place-items:center;height:100vh;margin:0}
main{max-width:420px;padding:24px;text-align:center}code{background:#1b1f27;padding:2px 6px;border-radius:4px}</style></head>
<body><main><h1>&#128274; Access token required</h1>
<p>This server is protected. Open it once with <code>?token=YOUR_TOKEN</code> appended to the URL.</p></main></body></html>"""


def _configure_logging() -> None:
    if not log.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s  %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        log.addHandler(handler)
        log.setLevel(logging.INFO)
        log.propagate = False


def lan_addresses() -> list[str]:
    """Best-effort list of this machine's IPv4 addresses on the local network."""
    ips: set[str] = set()
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("10.255.255.255", 1))  # no packets are sent for UDP connect
            ips.add(s.getsockname()[0])
    except OSError:
        pass
    try:
        for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET):
            ips.add(str(info[4][0]))
    except OSError:
        pass
    return sorted(ip for ip in ips if not ip.startswith(("127.", "169.254.")))


def create_app(settings: Settings | None = None, detector: BaseDetector | None = None) -> FastAPI:
    _configure_logging()
    settings = settings or Settings()
    db = Database(settings.db_path)
    detector = detector or HFDetector(settings)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        db.init()
        log.info("Database: %s", settings.db_path)
        detector.start_loading()
        urls = [f"http://{ip}:{settings.port}" for ip in lan_addresses()]
        log.info("Local:   http://localhost:%s", settings.port)
        for url in urls:
            log.info("Network: %s   <- open this on other devices on your Wi-Fi", url)
        if settings.access_token:
            log.info("Access token is ON — first visit must use  <url>/?token=<AIDET_ACCESS_TOKEN>")
        yield

    app = FastAPI(title="ScribeScan", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.state.db = db
    app.state.detector = detector

    # -- optional shared-secret gate for LAN use -------------------------------
    if settings.access_token:
        expected = settings.access_token.encode()

        def _valid(value: str | None) -> bool:
            return bool(value) and secrets.compare_digest(value.encode(), expected)

        @app.middleware("http")
        async def token_gate(request: Request, call_next):
            supplied = request.query_params.get("token")
            if _valid(supplied):
                clean = request.url.remove_query_params("token")
                resp = RedirectResponse(str(clean), status_code=303)
                resp.set_cookie(TOKEN_COOKIE, settings.access_token, httponly=True,
                                samesite="strict", max_age=60 * 60 * 24 * 365)
                return resp
            if _valid(request.cookies.get(TOKEN_COOKIE)) or _valid(request.headers.get("x-access-token")):
                return await call_next(request)
            if request.url.path.startswith("/api/"):
                return JSONResponse({"detail": "Access token required"}, status_code=401)
            return HTMLResponse(LOCKED_PAGE, status_code=401)

    # -- helpers -----------------------------------------------------------------
    def _check_size(payload: DocumentUpdate) -> None:
        size = len(payload.content.encode()) + len(payload.plain_text.encode())
        if size > settings.max_content_bytes:
            raise HTTPException(413,
                                f"Document exceeds {settings.max_content_bytes:,} bytes")

    def _get_or_404(doc_id: int) -> dict[str, Any]:
        doc = db.get_document(doc_id)
        if doc is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
        return doc

    # -- API: meta -----------------------------------------------------------------
    @app.get("/api/health")
    def health() -> dict[str, Any]:
        return {
            "status": "ok",
            "version": __version__,
            "model": detector.status(),
            "min_words": settings.min_words,
            "max_words": settings.max_words,
            "chunk_words": settings.chunk_words,
            "chunk_overlap": settings.chunk_overlap,
            "human_threshold": settings.human_threshold,
        }

    # -- API: documents ---------------------------------------------------------
    @app.get("/api/documents")
    def list_documents() -> list[dict[str, Any]]:
        return db.list_documents()

    @app.post("/api/documents", status_code=status.HTTP_201_CREATED)
    def create_document(payload: DocumentCreate | None = None) -> dict[str, Any]:
        title = (payload.title if payload else "").strip() or "Untitled document"
        return db.create_document(title)

    @app.get("/api/documents/{doc_id}")
    def get_document(doc_id: int) -> dict[str, Any]:
        doc = _get_or_404(doc_id)
        doc["latest_scan"] = db.latest_scan(doc_id, doc["plain_text"])
        return doc

    @app.put("/api/documents/{doc_id}")
    def update_document(doc_id: int, payload: DocumentUpdate):
        _check_size(payload)
        try:
            saved = db.update_document(
                doc_id,
                title=payload.title.strip() or "Untitled document",
                content=payload.content,
                plain_text=payload.plain_text,
                base_version=payload.version,
                force=payload.force,
            )
        except VersionConflict as conflict:
            return JSONResponse(
                {"detail": "This document was changed on another device.", "current": conflict.current},
                status_code=status.HTTP_409_CONFLICT,
            )
        if saved is None:
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
        return saved

    @app.delete("/api/documents/{doc_id}", status_code=status.HTTP_204_NO_CONTENT)
    def delete_document(doc_id: int) -> Response:
        if not db.delete_document(doc_id):
            raise HTTPException(status.HTTP_404_NOT_FOUND, "Document not found")
        return Response(status_code=status.HTTP_204_NO_CONTENT)

    # -- API: detection -----------------------------------------------------------
    # A plain `def` endpoint: FastAPI runs it in a worker thread, so GPU inference
    # never blocks autosaves from other tabs/devices.
    @app.post("/api/scan")
    def scan(payload: ScanRequest):
        try:
            result = detector.analyze(payload.text)
        except TextTooShort as e:
            return JSONResponse(
                {"detail": str(e), "code": "too_short", "word_count": e.word_count, "min_words": e.min_words},
                status_code=422,
            )
        except TextTooLong as e:
            return JSONResponse(
                {"detail": str(e), "code": "too_long", "word_count": e.word_count, "max_words": e.max_words},
                status_code=413,
            )
        except ModelNotReady:
            model = detector.status()
            msg = ("The model failed to load: " + str(model.get("error"))) if model["state"] == "error" \
                else "The model is still loading — try again in a few seconds."
            return JSONResponse({"detail": msg, "code": "model_not_ready", "model": model},
                                status_code=status.HTTP_503_SERVICE_UNAVAILABLE)

        if payload.document_id is not None:
            db.save_scan(payload.document_id, payload.text, result)
        return result

    # -- frontend -------------------------------------------------------------------
    @app.middleware("http")
    async def revalidate_static(request: Request, call_next):
        # Browsers must re-check (cheap 304) so devices never run a stale app.js/CSS after an update.
        response = await call_next(request)
        if request.url.path.startswith("/static/"):
            response.headers.setdefault("Cache-Control", "no-cache")
        return response

    app.mount("/static", StaticFiles(directory=settings.static_dir), name="static")

    @app.get("/", include_in_schema=False)
    def index() -> FileResponse:
        return FileResponse(settings.static_dir / "index.html", headers={"Cache-Control": "no-cache"})

    return app


app = create_app()
