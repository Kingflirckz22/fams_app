from sqlalchemy import Column, Integer, String, Boolean, DateTime, ForeignKey, Text, LargeBinary
from sqlalchemy.orm import relationship
from datetime import datetime
from .database import Base


class Student(Base):
    __tablename__ = "students"
    id = Column(Integer, primary_key=True, index=True)
    full_name = Column(String, nullable=False)
    registration_number = Column(String, unique=True, index=True, nullable=False)
    email = Column(String, unique=True, index=True, nullable=False)
    password_hash = Column(String, nullable=False)
    level = Column(String, nullable=True)  # ND1, ND2, HND1, HND2

    face_enrolled = Column(Boolean, default=False)
    # Embedding is stored directly in the database as a JSON string of
    # floats (not as a file on local disk). This survives server restarts
    # and cloud redeploys, unlike the old embeddings/*.pkl file approach.
    face_embedding = Column(Text, nullable=True)
    # A representative face image captured during enrollment, used as the
    # default profile picture unless the student uploads their own.
    profile_picture = Column(LargeBinary, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    attendance_records = relationship("Attendance", back_populates="student")
    course_enrollments = relationship("CourseEnrollment", back_populates="student")


class Lecturer(Base):
    __tablename__ = "lecturers"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True, nullable=False)
    full_name = Column(String, nullable=False)
    email = Column(String, unique=True, nullable=False)
    password_hash = Column(String, nullable=False)
    profile_picture = Column(LargeBinary, nullable=True)

    created_at = Column(DateTime, default=datetime.utcnow)
    courses = relationship("Course", back_populates="lecturer")
    sessions = relationship("Session", back_populates="lecturer")


class Course(Base):
    __tablename__ = "courses"
    id = Column(Integer, primary_key=True, index=True)
    course_name = Column(String, nullable=False)
    course_code = Column(String, nullable=False)
    lecturer_id = Column(Integer, ForeignKey("lecturers.id"))
    created_at = Column(DateTime, default=datetime.utcnow)
    lecturer = relationship("Lecturer", back_populates="courses")
    sessions = relationship("Session", back_populates="course")
    enrollments = relationship("CourseEnrollment", back_populates="course")


class CourseEnrollment(Base):
    __tablename__ = "course_enrollments"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    course_id = Column(Integer, ForeignKey("courses.id"))
    student = relationship("Student", back_populates="course_enrollments")
    course = relationship("Course", back_populates="enrollments")


class Session(Base):
    __tablename__ = "sessions"
    id = Column(Integer, primary_key=True, index=True)
    course_id = Column(Integer, ForeignKey("courses.id"))
    lecturer_id = Column(Integer, ForeignKey("lecturers.id"))
    date = Column(String, nullable=False)
    allowed_ip_range = Column(String, nullable=False)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    course = relationship("Course", back_populates="sessions")
    lecturer = relationship("Lecturer", back_populates="sessions")
    attendance_records = relationship("Attendance", back_populates="session")


class Attendance(Base):
    __tablename__ = "attendance"
    id = Column(Integer, primary_key=True, index=True)
    student_id = Column(Integer, ForeignKey("students.id"))
    session_id = Column(Integer, ForeignKey("sessions.id"))
    status = Column(String, default="Present")
    timestamp = Column(DateTime, default=datetime.utcnow)
    student = relationship("Student", back_populates="attendance_records")
    session = relationship("Session", back_populates="attendance_records")