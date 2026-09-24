from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from pydantic import BaseModel
import bcrypt

from .database import get_db
from .models import User


router = APIRouter(
    prefix="/api/auth",
    tags=["Authentication"]
)


# =========================
# REQUEST MODELS
# =========================

class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


# =========================
# REGISTER
# =========================

@router.post("/register")
def register(
    data: RegisterRequest,
    db: Session = Depends(get_db)
):

    name = data.name.strip()
    email = data.email.strip().lower()
    password = data.password


    # Validate input

    if not name:
        raise HTTPException(
            status_code=400,
            detail="Name is required."
        )


    if not email:
        raise HTTPException(
            status_code=400,
            detail="Email is required."
        )


    if len(password) < 6:
        raise HTTPException(
            status_code=400,
            detail="Password must be at least 6 characters."
        )


    # Check existing user

    existing_user = db.query(User).filter(
        User.email == email
    ).first()


    if existing_user:

        raise HTTPException(
            status_code=400,
            detail="Email already registered."
        )


    # Hash password

    hashed_password = bcrypt.hashpw(
        password.encode("utf-8"),
        bcrypt.gensalt()
    ).decode("utf-8")


    # Create user

    new_user = User(
        name=name,
        email=email,
        password=hashed_password
    )


    db.add(new_user)

    db.commit()

    db.refresh(new_user)


    return {
        "success": True,
        "message": "Registration successful.",
        "user": {
            "id": new_user.id,
            "name": new_user.name,
            "email": new_user.email
        }
    }


# =========================
# LOGIN
# =========================

@router.post("/login")
def login(
    data: LoginRequest,
    db: Session = Depends(get_db)
):

    email = data.email.strip().lower()
    password = data.password


    # Find user

    user = db.query(User).filter(
        User.email == email
    ).first()


    if not user:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )


    # Verify password

    password_valid = bcrypt.checkpw(
        password.encode("utf-8"),
        user.password.encode("utf-8")
    )


    if not password_valid:

        raise HTTPException(
            status_code=401,
            detail="Invalid email or password."
        )


    return {
        "success": True,
        "message": "Login successful.",
        "user": {
            "id": user.id,
            "name": user.name,
            "email": user.email
        }
    }