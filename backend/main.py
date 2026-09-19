from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse, FileResponse
from pathlib import Path
from .database import engine, Base
from .routes import students, lecturers, attendance
import os

Base.metadata.create_all(bind=engine)

app = FastAPI(title="FAMS — Facial Attendance Management System", version="1.0")

ALLOWED_ORIGINS = os.getenv("ALLOWED_ORIGINS", "*").split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register all route groups
app.include_router(students.router)
app.include_router(lecturers.router)
app.include_router(attendance.router)

FRONTEND_DIR = Path("frontend").resolve()


def _safe_resolve(relative_path: str) -> Path:
    """Resolves a path inside frontend/ and blocks any attempt to escape
    the folder (e.g. someone requesting /app/../backend/main.py)."""
    candidate = (FRONTEND_DIR / relative_path).resolve()
    if not str(candidate).startswith(str(FRONTEND_DIR)):
        raise HTTPException(status_code=404, detail="Not Found")
    return candidate


@app.get("/app")
def app_root():
    return RedirectResponse(url="/app/index.html")


@app.get("/app/{path:path}")
def serve_frontend(path: str):
    if path in ("", "/"):
        target = _safe_resolve("index.html")
        if target.is_file():
            return FileResponse(target)
        raise HTTPException(status_code=404, detail="Not Found")

    exact = _safe_resolve(path)
    if exact.is_file():
        return FileResponse(exact)

    with_html = _safe_resolve(f"{path}.html")
    if with_html.is_file():
        return FileResponse(with_html)

    raise HTTPException(status_code=404, detail="Not Found")


@app.get("/")
def root():
    return RedirectResponse(url="/app/index.html")