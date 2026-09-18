import os
import bcrypt
from jose import JWTError, jwt
from datetime import datetime, timedelta
from fastapi import Depends, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials

SECRET_KEY = os.getenv("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY environment variable is not set. "
        "Set it in your .env file (local) or your hosting platform's "
        "environment variables (production) before starting the server. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_HOURS = 24

security = HTTPBearer()

def hash_password(password: str) -> str:
    # bcrypt requires bytes and has a 72-byte limit
    pw_bytes = password.encode("utf-8")[:72]
    hashed = bcrypt.hashpw(pw_bytes, bcrypt.gensalt())
    return hashed.decode("utf-8")

def verify_password(plain: str, hashed: str) -> bool:
    pw_bytes = plain.encode("utf-8")[:72]
    return bcrypt.checkpw(pw_bytes, hashed.encode("utf-8"))

def create_token(data: dict) -> str:
    to_encode = data.copy()
    to_encode["exp"] = datetime.utcnow() + timedelta(hours=ACCESS_TOKEN_EXPIRE_HOURS)
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def decode_token(token: str) -> dict:
    try:
        return jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

def get_current_student(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = decode_token(credentials.credentials)
    if payload.get("role") != "student":
        raise HTTPException(status_code=403, detail="Student access required")
    return payload

def get_current_lecturer(credentials: HTTPAuthorizationCredentials = Depends(security)):
    payload = decode_token(credentials.credentials)
    if payload.get("role") != "lecturer":
        raise HTTPException(status_code=403, detail="Lecturer access required")
    return payload