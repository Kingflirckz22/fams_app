from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request, Response
from sqlalchemy.orm import Session
from datetime import datetime
from ..database import get_db
from ..models import Student, Attendance, Session as AttSession, Course, CourseEnrollment
from ..auth import hash_password, verify_password, create_token, get_current_student
from ..services.face_service import (
    process_video_for_enrollment,
    compare_face_to_stored,
    extract_face_jpeg_from_image,
    compute_similarity,
    MATCH_THRESHOLD
)
import cv2
import numpy as np
import tempfile
import os

router = APIRouter(prefix="/students", tags=["students"])

VALID_LEVELS = ["ND1", "ND2", "HND1", "HND2"]


@router.post("/register")
def register_student(
    full_name: str = Form(...),
    registration_number: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    level: str = Form(None),
    db: Session = Depends(get_db)
):
    if db.query(Student).filter(Student.email == email).first():
        raise HTTPException(status_code=409, detail="Email already registered")
    if db.query(Student).filter(Student.registration_number == registration_number).first():
        raise HTTPException(status_code=409, detail="Registration number already exists")
    if level and level not in VALID_LEVELS:
        raise HTTPException(status_code=400, detail=f"Level must be one of {VALID_LEVELS}")

    student = Student(
        full_name=full_name,
        registration_number=registration_number,
        email=email,
        password_hash=hash_password(password),
        level=level
    )
    db.add(student)
    db.commit()
    db.refresh(student)
    return {"message": "Account created successfully", "student_id": student.id}


@router.post("/login")
def login_student(
    email: str = Form(...),
    password: str = Form(...),
    db: Session = Depends(get_db)
):
    student = db.query(Student).filter(Student.email == email).first()
    if not student or not verify_password(password, student.password_hash):
        raise HTTPException(status_code=401, detail="Invalid email or password")
    token = create_token({
        "sub": str(student.id),
        "email": student.email,
        "reg": student.registration_number,
        "name": student.full_name,
        "role": "student",
        "face_enrolled": student.face_enrolled
    })
    return {
        "token": token,
        "face_enrolled": student.face_enrolled,
        "name": student.full_name
    }


@router.post("/enroll-face")
async def enroll_face(
    video: UploadFile = File(...),
    current: dict = Depends(get_current_student),
    db: Session = Depends(get_db)
):
    student = db.query(Student).filter(Student.id == int(current["sub"])).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    suffix = os.path.splitext(video.filename)[1] or ".webm"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        content = await video.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        result = process_video_for_enrollment(tmp_path)
    finally:
        os.unlink(tmp_path)

    if not result["success"]:
        raise HTTPException(status_code=400, detail=result["message"])

    # Reject if this face is already enrolled under a different account.
    # Excludes the current student so re-enrolling your own face never
    # falsely flags against yourself.
    other_students = db.query(Student).filter(
        Student.id != student.id,
        Student.face_embedding.isnot(None)
    ).all()

    for other in other_students:
        similarity = compute_similarity(result["embedding_json"], other.face_embedding)
        if similarity >= MATCH_THRESHOLD:
            raise HTTPException(
                status_code=409,
                detail="This face appears to already be enrolled under a different student account. If you believe this is an error, contact your department admin."
            )

    # Embedding and representative face photo are stored directly in the
    # database — nothing is written to local disk. This survives server
    # restarts and cloud redeploys.
    student.face_enrolled = True
    student.face_embedding = result["embedding_json"]
    if result["profile_jpeg"] and not student.profile_picture:
        # Only auto-set as profile picture if the student hasn't already
        # uploaded their own photo manually.
        student.profile_picture = result["profile_jpeg"]
    db.commit()
    return {"message": result["message"], "frames_used": result.get("frames_used")}


@router.get("/dashboard")
def student_dashboard(
    current: dict = Depends(get_current_student),
    db: Session = Depends(get_db)
):
    student = db.query(Student).filter(Student.id == int(current["sub"])).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    enrollments = db.query(CourseEnrollment).filter(
        CourseEnrollment.student_id == student.id
    ).all()

    course_data = []
    for enrollment in enrollments:
        course = enrollment.course
        sessions = db.query(AttSession).filter(
            AttSession.course_id == course.id
        ).all()
        total_sessions = len(sessions)
        attended = 0
        session_details = []
        for sess in sessions:
            att = db.query(Attendance).filter(
                Attendance.student_id == student.id,
                Attendance.session_id == sess.id
            ).first()
            status = att.status if att else "Absent"
            time_marked = att.timestamp.strftime("%H:%M") if att else "—"
            if att:
                attended += 1
            session_details.append({
                "session_id": sess.id,
                "date": sess.date,
                "status": status,
                "time_marked": time_marked
            })
        course_data.append({
            "course_name": course.course_name,
            "course_code": course.course_code,
            "total_sessions": total_sessions,
            "attended": attended,
            "missed": total_sessions - attended,
            "percentage": round((attended / total_sessions * 100), 1) if total_sessions > 0 else 0,
            "sessions": session_details
        })

    return {
        "student_name": student.full_name,
        "registration_number": student.registration_number,
        "level": student.level,
        "has_profile_picture": student.profile_picture is not None,
        "courses": course_data
    }


@router.get("/active-sessions")
def active_sessions(
    current: dict = Depends(get_current_student),
    db: Session = Depends(get_db)
):
    student = db.query(Student).filter(Student.id == int(current["sub"])).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    if not student.face_enrolled:
        raise HTTPException(status_code=400, detail="Please enrol your face before viewing sessions.")

    active = db.query(AttSession).filter(AttSession.is_active == True).all()

    result = []
    for s in active:
        minutes_ago = int((datetime.utcnow() - s.created_at).total_seconds() // 60)
        result.append({
            "session_id": s.id,
            "course_name": s.course.course_name,
            "course_code": s.course.course_code,
            "lecturer_name": s.lecturer.full_name,
            "minutes_ago": max(minutes_ago, 0)
        })

    return {"sessions": result}


@router.put("/profile")
def update_profile(
    level: str = Form(...),
    current: dict = Depends(get_current_student),
    db: Session = Depends(get_db)
):
    """Lets a student change their level after registration (e.g. ND1 -> ND2)."""
    if level not in VALID_LEVELS:
        raise HTTPException(status_code=400, detail=f"Level must be one of {VALID_LEVELS}")
    student = db.query(Student).filter(Student.id == int(current["sub"])).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    student.level = level
    db.commit()
    return {"message": "Profile updated", "level": student.level}


@router.post("/profile-picture")
async def upload_profile_picture(
    photo: UploadFile = File(...),
    current: dict = Depends(get_current_student),
    db: Session = Depends(get_db)
):
    """Lets a student manually upload/replace their profile picture,
    independent of the auto-captured enrollment photo."""
    student = db.query(Student).filter(Student.id == int(current["sub"])).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")

    contents = await photo.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    face_jpeg = extract_face_jpeg_from_image(image)
    if face_jpeg is None:
        raise HTTPException(status_code=400, detail="No face detected in the uploaded photo")

    student.profile_picture = face_jpeg
    db.commit()
    return {"message": "Profile picture updated"}


@router.get("/photo")
def get_my_photo(
    current: dict = Depends(get_current_student),
    db: Session = Depends(get_db)
):
    """Serves the logged-in student's own profile picture as a JPEG."""
    student = db.query(Student).filter(Student.id == int(current["sub"])).first()
    if not student or not student.profile_picture:
        raise HTTPException(status_code=404, detail="No profile picture set")
    return Response(content=student.profile_picture, media_type="image/jpeg")