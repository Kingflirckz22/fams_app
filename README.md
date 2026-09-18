# FAMS — Facial Attendance Management System

A mobile-based classroom attendance system using deep learning facial recognition, liveness detection, and Wi-Fi proximity verification. Built as a final year project at Federal Polytechnic Ado-Ekiti, Department of Networking and Cloud Computing.

## Features

- **Facial enrolment & recognition** — students enrol via a short guided video capture; FaceNet-512 (via DeepFace) generates a facial embedding used for future attendance verification.
- **Liveness detection** — passive anti-spoofing checks (texture, sharpness, frequency analysis) prevent attendance being marked from a photo or screen recapture.
- **Wi-Fi proximity verification** — each attendance session is bound to a classroom/campus IP range, so attendance can only be marked by devices physically on that network.
- **Duplicate-face protection** — a face already enrolled under one student account cannot be enrolled again under a different account.
- **Lecturer tools** — course and session management, live attendance dashboards, and Excel attendance exports.
- **Role-based access** — separate student and lecturer portals, secured with JWT authentication.

## Tech stack

| Layer | Technology |
|---|---|
| Backend | FastAPI (Python), SQLAlchemy |
| Face detection | MediaPipe |
| Face recognition | FaceNet-512 via DeepFace |
| Database | PostgreSQL (Neon) |
| Frontend | HTML5, Tailwind CSS, vanilla JavaScript |
| Auth | JWT (python-jose), bcrypt |

## Project structure

```
backend/
  routes/         # API endpoints (students, lecturers, sessions, attendance)
  services/       # Face detection, recognition, and liveness logic
  main.py         # App entry point
  models.py       # Database models
  auth.py         # JWT & password handling
  database.py     # DB connection setup
frontend/         # Student and lecturer web portal (HTML/CSS/JS)
requirements.txt
```

## Running locally

1. Clone the repo and create a virtual environment
2. `pip install -r requirements.txt`
3. Create a `.env` file with `DATABASE_URL`, `SECRET_KEY`, `LECTURER_ACCESS_CODE` (see `backend/auth.py` and `backend/routes/lecturers.py` for how each is used and how to generate a secure value)
4. `uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000`
5. Open `http://localhost:8000/app/index.html`

## Author

Final year project — HND II, Computer Science, Federal Polytechnic Ado-Ekiti.