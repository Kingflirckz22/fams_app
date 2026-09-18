from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, Request
from sqlalchemy.orm import Session
from ..database import get_db
from ..models import Student, Attendance, Session as AttSession, CourseEnrollment
from ..auth import get_current_student
from ..services.face_service import compare_face_to_stored
import cv2
import numpy as np
import ipaddress

router = APIRouter(prefix="/attendance", tags=["attendance"])


def check_wifi(request: Request, allowed_range: str) -> bool:
    client_ip = request.client.host
    forwarded = request.headers.get("X-Forwarded-For")
    if forwarded:
        client_ip = forwarded.split(",")[0].strip()
    try:
        network = ipaddress.ip_network(allowed_range, strict=False)
        return ipaddress.ip_address(client_ip) in network
    except Exception:
        return False


@router.post("/mark")
async def mark_attendance(
    request: Request,
    session_id: int = Form(...),
    photo: UploadFile = File(...),
    current: dict = Depends(get_current_student),
    db: Session = Depends(get_db)
):
    sess = db.query(AttSession).filter(AttSession.id == session_id).first()
    if not sess:
        raise HTTPException(status_code=404, detail="Session not found")
    if not sess.is_active:
        raise HTTPException(status_code=400, detail="This session is no longer active")

    if not check_wifi(request, sess.allowed_ip_range):
        raise HTTPException(
            status_code=403,
            detail="You must be connected to the classroom Wi-Fi to mark attendance"
        )

    student = db.query(Student).filter(Student.id == int(current["sub"])).first()
    if not student:
        raise HTTPException(status_code=404, detail="Student not found")
    if not student.face_enrolled:
        raise HTTPException(status_code=400, detail="Face not enrolled. Please enrol your face first.")

    existing = db.query(Attendance).filter(
        Attendance.student_id == student.id,
        Attendance.session_id == session_id
    ).first()
    if existing:
        raise HTTPException(status_code=409, detail="Attendance already marked for this session")

    contents = await photo.read()
    np_arr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if image is None:
        raise HTTPException(status_code=400, detail="Could not read image")

    # student.face_embedding is now a JSON string stored directly in the
    # database (see models.py), not a file path — liveness check runs
    # first inside compare_face_to_stored, before any recognition.
    result = compare_face_to_stored(image, student.face_embedding)
    if not result["match"]:
        raise HTTPException(status_code=401, detail=result["message"])

    course_enrollment = db.query(CourseEnrollment).filter(
        CourseEnrollment.student_id == student.id,
        CourseEnrollment.course_id == sess.course_id
    ).first()
    if not course_enrollment:
        db.add(CourseEnrollment(student_id=student.id, course_id=sess.course_id))

    record = Attendance(
        student_id=student.id,
        session_id=session_id,
        status="Present"
    )
    db.add(record)
    db.commit()
    return {
        "message": "Attendance marked successfully",
        "student_name": student.full_name,
        "similarity": result.get("similarity"),
        "liveness_score": result.get("liveness_score")
    }