import logging
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
from fastapi import FastAPI, Depends, UploadFile, File, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from sqlalchemy.orm import Session

from .database import engine, get_db, Base
from .models import User, TranslationHistory
from .auth import router as auth_router
from .ml_combined import predict_frame
from .ml_jksy import predict_jksy

logger = logging.getLogger("signmate")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

# Initialize database schema
Base.metadata.create_all(bind=engine)

app = FastAPI(
    title="SignMate",
    description="Indian Sign Language Recognition & Two-Way Translation Platform",
    version="2.1.0"
)

app.include_router(auth_router)

BASE_DIR = Path(__file__).resolve().parent.parent

# Serve frontend static assets
app.mount(
    "/static",
    StaticFiles(directory=BASE_DIR / "frontend"),
    name="static"
)


@app.get("/")
def home():
    """Serves the single-page application frontend."""
    return FileResponse(BASE_DIR / "frontend" / "index.html")


@app.get("/api/status")
def api_status():
    """Healthcheck endpoint reporting active capabilities."""
    return {
        "application": "SignMate",
        "status": "online",
        "message": "SignMate Real-Time Engine is active.",
        "version": "2.1.0",
        "features": [
            "sign-to-text",
            "speech-to-isl",
            "text-to-isl",
            "isl-dictionary",
            "history"
        ]
    }


@app.get("/api/database-test")
def database_test(db: Session = Depends(get_db)):
    """Database connectivity verification."""
    user_count = db.query(User).count()
    history_count = db.query(TranslationHistory).count()
    return {
        "database": "SQLite",
        "status": "connected",
        "users": user_count,
        "translation_history": history_count
    }


@app.post("/api/translate/sign")
async def translate_sign(file: UploadFile = File(...)):
    """Processes an uploaded webcam frame through the multi-model arbitration pipeline."""
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image payload received.")

        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Could not decode image.")

        result = predict_frame(frame)
        return result

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Sign translation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unable to process sign gesture.")


@app.post("/api/translate/jksy")
async def translate_jksy(file: UploadFile = File(...)):
    """Specialist classification endpoint for J, K, S, and Y signs."""
    try:
        image_bytes = await file.read()
        if not image_bytes:
            raise HTTPException(status_code=400, detail="Empty image received.")

        image_array = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image payload.")

        label, confidence = predict_jksy(frame)
        return {
            "success": label is not None,
            "label": label,
            "confidence": confidence,
            "message": "J/K/S/Y specialist prediction."
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"JKSY translation error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Unable to process gesture.")


class HistoryRequest(BaseModel):
    user_id: int
    mode: str
    input_text: Optional[str] = None
    output_text: Optional[str] = None


@app.post("/api/history")
def save_history(data: HistoryRequest, db: Session = Depends(get_db)):
    """Saves a translation record for an authenticated user."""
    user = db.query(User).filter(User.id == data.user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="User account not found.")

    history = TranslationHistory(
        user_id=data.user_id,
        mode=data.mode,
        input_text=data.input_text,
        output_text=data.output_text
    )
    db.add(history)
    db.commit()
    db.refresh(history)

    return {
        "success": True,
        "message": "Translation saved to history.",
        "id": history.id
    }


@app.get("/api/history/{user_id}")
def get_history(user_id: int, db: Session = Depends(get_db)):
    """Retrieves the recent translation history for a user."""
    history = (
        db.query(TranslationHistory)
        .filter(TranslationHistory.user_id == user_id)
        .order_by(TranslationHistory.created_at.desc())
        .limit(50)
        .all()
    )

    return {
        "success": True,
        "history": [
            {
                "id": item.id,
                "mode": item.mode,
                "input_text": item.input_text,
                "output_text": item.output_text,
                "created_at": item.created_at.isoformat() if item.created_at else None
            }
            for item in history
        ]
    }


@app.delete("/api/history/{history_id}")
def delete_history_item(history_id: int, db: Session = Depends(get_db)):
    """Deletes a single history record by ID."""
    item = db.query(TranslationHistory).filter(TranslationHistory.id == history_id).first()
    if not item:
        raise HTTPException(status_code=404, detail="History record not found.")

    db.delete(item)
    db.commit()
    return {"success": True, "message": "History item removed."}


@app.delete("/api/history/user/{user_id}")
def clear_user_history(user_id: int, db: Session = Depends(get_db)):
    """Clears all history records for a user."""
    count = db.query(TranslationHistory).filter(TranslationHistory.user_id == user_id).delete()
    db.commit()
    return {"success": True, "message": f"Cleared {count} history records."}


# Standard Indian Sign Language 26 Alphabet Definitions
ISL_ALPHABET_DATA = [
    {"letter": "A", "type": "Two-Hand", "desc": "Both thumbs touch at the tips to form an 'A' peak, other fingers curled."},
    {"letter": "B", "type": "Two-Hand", "desc": "Join index and thumb of both hands together to form two linked loops like 'B'."},
    {"letter": "C", "type": "One-Hand", "desc": "Curve active hand fingers and thumb to form a clear 'C' shape."},
    {"letter": "D", "type": "Two-Hand", "desc": "Non-dominant index points vertically; dominant index & thumb form a semi-circle loop meeting it."},
    {"letter": "E", "type": "Two-Hand", "desc": "Extend index fingers horizontally pointing straight toward each other."},
    {"letter": "F", "type": "Two-Hand", "desc": "Cross the dominant index and middle fingers flat over the non-dominant index and middle fingers."},
    {"letter": "G", "type": "Two-Hand", "desc": "Both hands clenched in fists, placing one fist directly on top of the other."},
    {"letter": "H", "type": "Two-Hand", "desc": "Flat dominant palm strokes across the open palm/back of the non-dominant hand."},
    {"letter": "I", "type": "One-Hand", "desc": "Dominant hand closed in a fist with pinky finger held straight upright."},
    {"letter": "J", "type": "Two-Hand", "desc": "Draw a hook/curve letter 'J' using dominant index finger onto non-dominant palm."},
    {"letter": "K", "type": "Two-Hand", "desc": "Left index finger held straight up; right index hooks against knuckle to form the angle of 'K'."},
    {"letter": "L", "type": "One-Hand", "desc": "Extend thumb and index finger outward at a 90° angle forming the letter 'L'."},
    {"letter": "M", "type": "Two-Hand", "desc": "Three fingers (index, middle, ring) of dominant hand rest across the flat non-dominant palm."},
    {"letter": "N", "type": "Two-Hand", "desc": "Two fingers (index and middle) of dominant hand rest across the flat non-dominant palm."},
    {"letter": "O", "type": "One-Hand", "desc": "Touch tips of thumb and all fingers together forming an 'O' circle."},
    {"letter": "P", "type": "Two-Hand", "desc": "Form a circle with dominant thumb & index, resting against the upright non-dominant index finger."},
    {"letter": "Q", "type": "Two-Hand", "desc": "Hook dominant index finger inside the circle of non-dominant thumb and index."},
    {"letter": "R", "type": "Two-Hand", "desc": "Dominant hooked index finger sits upright in the palm of the flat non-dominant hand."},
    {"letter": "S", "type": "Two-Hand", "desc": "Both pinky fingers hook securely together."},
    {"letter": "T", "type": "Two-Hand", "desc": "Dominant index finger taps the edge of the open non-dominant hand below the thumb."},
    {"letter": "U", "type": "One-Hand", "desc": "Index and middle fingers held together straight up with thumb securing ring & pinky."},
    {"letter": "V", "type": "One-Hand", "desc": "Index and middle fingers held up and spread apart in a clear 'V' sign."},
    {"letter": "W", "type": "Two-Hand", "desc": "Fingers of both hands interlock and point upward/outward like the ridges of 'W'."},
    {"letter": "X", "type": "Two-Hand", "desc": "Cross dominant and non-dominant index fingers to form an 'X'."},
    {"letter": "Y", "type": "Two-Hand", "desc": "Dominant index finger placed into the V-cleft between the non-dominant thumb and index finger."},
    {"letter": "Z", "type": "Two-Hand", "desc": "One flat hand placed bent against the perpendicular flat palm to form a 'Z' silhouette."}
]


@app.get("/api/dictionary")
def get_isl_dictionary():
    """Returns the 26 standard ISL alphabet signs and metadata."""
    return {
        "success": True,
        "alphabet_count": len(ISL_ALPHABET_DATA),
        "words_count": 0,
        "alphabet": [
            {
                **item,
                "image_jpg": f"/static/isl/{item['letter']}.jpg",
                "image_png": f"/static/isl/{item['letter']}.png"
            }
            for item in ISL_ALPHABET_DATA
        ],
        "words": []
    }