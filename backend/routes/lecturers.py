from fastapi import APIRouter, Depends, HTTPException, Form, Request, UploadFile, File, Response
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Lecturer, Course, CourseEnrollment, Student, Session as AttSession, Attendance
from ..auth import hash_password, verify_password, create_token, get_current_lecturer
from ..services.face_service import extract_face_jpeg_from_image
import pandas as pd
from fastapi.responses import FileResponse
import os
import cv2
import numpy as np
from datetime import datetime

router = APIRouter(prefix="/lecturers", tags=["lecturers"])

LECTURER_ACCESS_CODE = os.getenv("LECTURER_ACCESS_CODE")
if not LECTURER_ACCESS_CODE:
    raise RuntimeError(
        "LECTURER_ACCESS_CODE environment variable is not set. "
        "Set it in your .env file (local) or your hosting platform's "
        "environment variables (production) before starting the server."
    )


@router.post("/register")
def register_lecturer(
    full_name: str = Form(...),
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    staff_code: str = Form(...),
    db: Session = Depends(get_db)
):
    if staff_code != LECTURER_ACCESS_CODE:
        raise HTTPException(status_code=403, detail="Invalid staff access code. Contact your department admin.")

    if db.query(Lecturer).filter(Lecturer.username == username).first():
        raise HTTPException(status_code=409, detail="Username already taken")
    if db.query(Lecturer).filter(Lecturer.email == email).first():
        raise HTTPException(status_code=409, detail="Email already registered")

    lec = Lecturer(
        full_name=full_name,
        username=username,
        email=email,
        password_hash=hash_password(password)
    )
    db.add(lec)
    db.commit()
    return {"message": "Lecturer account created"}


@router.post("/login")
def login_lecturer(
    username: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    lec = db.query(Lecturer).filter(Lecturer.username == username).first()
    if not lec or not verify_password(password, lec.password_hash):
        raise HTTPException(status_code=401, detail="Invalid credentials")
    token = create_token({"sub": str(lec.id), "name": lec.full_name, "role": "lecturer"})
    return {"token": token, "name": lec.full_name}


@router.get("/detect-network")
def detect_network(
    request: Request,
    current: dict = Depends(get_current_lecturer)
):
    """
    Returns the IP address the lecturer's own device is currently seen
    as by the server, plus a suggested /24 range built from it.

    Testing phase: run this while your phone/laptop is on your own home
    Wi-Fi — it detects your current LAN IP (e.g. 192.168.0.42) and
    suggests 192.168.0.0/24, accepting any device on that same network.

    Production: run this from a device on the actual classroom/campus
    Wi-Fi at lecture time. If the whole campus shares one public IP via
    NAT, a /32 (single address) may be more accurate — the lecturer can
    edit the detected value before saving.
    """
    client_ip = request.client.host
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()

    suggested_range = None
    try:
        if client_ip and client_ip.count(".") == 3:
            parts = client_ip.split(".")
            suggested_range = f"{parts[0]}.{parts[1]}.{parts[2]}.0/24"
        elif client_ip:
            suggested_range = f"{client_ip}/32"
    except Exception:
        suggested_range = None

    return {"your_ip": client_ip, "suggested_range": suggested_range}


@router.get("/courses")
def list_my_courses(
    current: dict = Depends(get_current_lecturer),
    db: Session = Depends(get_db)
):
    courses = db.query(Course).filter(Course.lecturer_id == int(current["sub"])).all()
    return {
        "courses": [
            {"id": c.id, "course_name": c.course_name, "course_code": c.course_code}
            for c in courses
        ]
    }


@router.post("/courses/create")
def create_course(
    course_name: str = Form(...),
    course_code: str = Form(...),
    current: dict = Depends(get_current_lecturer),
    db: Session = Depends(get_db)
):
    course = Course(
        course_name=course_name,
        course_code=course_code,
        lecturer_id=int(current["sub"])
    )
    db.add(course)
    db.commit()
    db.refresh(course)
    return {"message": "Course created", "course_id": course.id}


@router.post("/courses/{course_id}/enroll")
def enroll_students(
    course_id: int,
    registration_numbers: str = Form(...),
    current: dict = Depends(get_current_lecturer),
    db: Session = Depends(get_db)
):
    reg_list = [r.strip() for r in registration_numbers.split(",")]
    enrolled = []
    not_found = []
    for reg in reg_list:
        student = db.query(Student).filter(Student.registration_number == reg).first()
        if not student:
            not_found.append(reg)
            continue
        existing = db.query(CourseEnrollment).filter(
            CourseEnrollment.student_id == student.id,
            CourseEnrollment.course_id == course_id
        ).first()
        if not existing:
            db.add(CourseEnrollment(student_id=student.id, course_id=course_id))
            enrolled.append(reg)
    db.commit()
    return {"enrolled": enrolled, "not_found": not_found}


@router.get("/sessions")
def list_my_sessions(
    current: dict = Depends(get_current_lecturer),
    db: Session = Depends(get_db)
):
    sessions = db.query(AttSession).filter(
        AttSession.lecturer_id == int(current["sub"])
    ).order_by(AttSession.created_at.desc()).all()

    result = []
    for s in sessions:
        total_enrolled = db.query(CourseEnrollment).filter(
            CourseEnrollment.course_id == s.course_id
        ).count()
        present_count = db.query(Attendance).filter(
            Attendance.session_id == s.id,
            Attendance.status == "Present"
        ).count()
        result.append({
            "id": s.id,
            "course_id": s.course_id,
            "course_name": s.course.course_name,
            "course_code": s.course.course_code,
            "date": s.date,
            "is_active": s.is_active,
            "total_enrolled": total_enrolled,
            "present_count": present_count
        })
    return {"sessions": result}


@router.post("/sessions/create")
def create_session(
    course_id: int = Form(...),
    date: str = Form(...),
    allowed_ip_range: str = Form(...),
    current: dict = Depends(get_current_lecturer),
    db: Session = Depends(get_db)
):
    sess = AttSession(
        course_id=course_id,
        lecturer_id=int(current["sub"]),
        date=date,
        allowed_ip_range=allowed_ip_range
    )
    db.add(sess)
    db.commit()
    db.refresh(sess)
    return {"message": "Session created", "session_id": sess.id}


@router.post("/sessions/{session_id}/end")
def end_session(
    session_id: int,
    current: dict = Depends(get_current_lecturer),
    db: Session = Depends(get_db)
):
    sess = db.query(AttSession).filter(AttSession.id == session_id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    if sess.lecturer_id != int(current["sub"]):
        raise HTTPException(status_code=403, detail="You do not own this session")
    sess.is_active = False
    db.commit()
    return {"message": "Session ended"}


@router.get("/sessions/{session_id}/attendance")
def view_attendance(
    session_id: int,
    current: dict = Depends(get_current_lecturer),
    db: Session = Depends(get_db)
):
    sess = db.query(AttSession).filter(AttSession.id == session_id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    enrollments = db.query(CourseEnrollment).filter(
        CourseEnrollment.course_id == sess.course_id
    ).all()
    result = []
    for e in enrollments:
        student = e.student
        att = db.query(Attendance).filter(
            Attendance.student_id == student.id,
            Attendance.session_id == session_id
        ).first()
        result.append({
            "registration_number": student.registration_number,
            "full_name": student.full_name,
            "status": att.status if att else "Absent",
            "time_marked": att.timestamp.strftime("%H:%M:%S") if att else "—"
        })
    return {"session_id": session_id, "date": sess.date, "attendance": result}


@router.get("/sessions/{session_id}/export")
def export_excel(
    session_id: int,
    current: dict = Depends(get_current_lecturer),
    db: Session = Depends(get_db)
):
    sess = db.query(AttSession).filter(AttSession.id == session_id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    enrollments = db.query(CourseEnrollment).filter(
        CourseEnrollment.course_id == sess.course_id
    ).all()
    rows = []
    for e in enrollments:
        student = e.student
        att = db.query(Attendance).filter(
            Attendance.student_id == student.id,
            Attendance.session_id == session_id
        ).first()
        rows.append({
            "Registration Number": student.registration_number,
            "Full Name": student.full_name,
            "Date": sess.date,
            "Time Marked": att.timestamp.strftime("%H:%M:%S") if att else "—",
            "Status": att.status if att else "Absent",
            "Present ✓": "✓" if att else ""
        })
    df = pd.DataFrame(rows)
    os.makedirs("attendance_files", exist_ok=True)
    filename = f"attendance_files/Attendance_{sess.course.course_code}_{sess.date}.xlsx"
    df.to_excel(filename, index=False)
    return FileResponse(
        filename,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        filename=os.path.basename(filename)
    )


@router.post("/profile-picture")
async def upload_lecturer_profile_picture(
    photo: UploadFile = File(...),
    current: dict = Depends(get_current_lecturer),
    db: Session = Depends(get_db)
):
    lec = db.query(Lecturer).filter(Lecturer.id == int(current["sub"])).first()
    if not lec:
        raise HTTPException(status_code=404, detail="Lecturer not found")

    contents = await photo.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    face_jpeg = extract_face_jpeg_from_image(image)
    if face_jpeg is None:
        raise HTTPException(status_code=400, detail="No face detected in the uploaded photo")

    lec.profile_picture = face_jpeg
    db.commit()
    return {"message": "Profile picture updated"}


@router.get("/photo")
def get_my_photo(
    current: dict = Depends(get_current_lecturer),
    db: Session = Depends(get_db)
):
    lec = db.query(Lecturer).filter(Lecturer.id == int(current["sub"])).first()
    if not lec or not lec.profile_picture:
        raise HTTPException(status_code=404, detail="No profile picture set")
    return Response(content=lec.profile_picture, media_type="image/jpeg")