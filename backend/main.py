from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from .database import engine, Base
from .routes import students, lecturers, attendance
import os

# Create all database tables
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

# Serve the frontend HTML files at /app
app.mount("/app", StaticFiles(directory="frontend", html=True), name="frontend")

from fastapi.responses import RedirectResponse

@app.get("/")
def root():
    return RedirectResponse(url="/app/index.html")