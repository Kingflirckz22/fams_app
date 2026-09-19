import cv2
import numpy as np
import json
import os
import mediapipe as mp
from deepface import DeepFace
import tempfile

mp_face = mp.solutions.face_detection

MATCH_THRESHOLD = 0.6           # same-person verification (attendance marking)
DUPLICATE_THRESHOLD = 0.78      # different-person duplicate detection (enrolment)
# Duplicate-check needs a stricter (higher) bar than same-person matching:
# wrongly blocking a real student from enrolling is a worse outcome than
# occasionally missing a genuine duplicate-account attempt.


def detect_and_crop_face(image_array: np.ndarray):
    """Detect face in image array and return cropped face."""
    with mp_face.FaceDetection(min_detection_confidence=0.6) as detector:
        rgb = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        results = detector.process(rgb)
        if not results.detections:
            return None
        detection = results.detections[0]
        bbox = detection.location_data.relative_bounding_box
        h, w = image_array.shape[:2]
        x = max(0, int(bbox.xmin * w))
        y = max(0, int(bbox.ymin * h))
        bw = int(bbox.width * w)
        bh = int(bbox.height * h)
        pad_x = int(bw * 0.2)
        pad_y = int(bh * 0.2)
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)
        cropped = image_array[y1:y2, x1:x2]
        return cropped if cropped.size > 0 else None


def generate_embedding(face_image: np.ndarray) -> list:
    """Generate FaceNet-512 embedding from face image array."""
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
        cv2.imwrite(tmp.name, face_image)
        tmp_path = tmp.name
    try:
        result = DeepFace.represent(
            img_path=tmp_path,
            model_name="Facenet512",
            enforce_detection=False,
            detector_backend="skip"
        )
        return result[0]["embedding"]
    finally:
        os.unlink(tmp_path)


def compute_similarity(embedding_a_json: str, embedding_b_json: str) -> float:
    """Cosine similarity between two stored embeddings (both JSON strings)."""
    a = np.array(json.loads(embedding_a_json))
    b = np.array(json.loads(embedding_b_json))
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


# LIVENESS / ANTI-SPOOFING CHECK (unchanged from previous version)

def _compute_lbp(gray: np.ndarray) -> np.ndarray:
    h, w = gray.shape
    lbp = np.zeros((h - 2, w - 2), dtype=np.uint8)
    center = gray[1:h-1, 1:w-1]
    offsets = [(-1,-1),(-1,0),(-1,1),(0,1),(1,1),(1,0),(1,-1),(0,-1)]
    for i, (dy, dx) in enumerate(offsets):
        neighbor = gray[1+dy:h-1+dy, 1+dx:w-1+dx]
        lbp |= ((neighbor >= center).astype(np.uint8) << i)
    return lbp


def check_liveness(face_image: np.ndarray) -> dict:
    if face_image is None or face_image.size == 0:
        return {"live": False, "score": 0.0, "reason": "No face region to analyse"}

    gray = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, (160, 160))

    lbp = _compute_lbp(gray)
    hist, _ = np.histogram(lbp.ravel(), bins=256, range=(0, 256), density=True)
    hist = hist[hist > 0]
    lbp_entropy = float(-np.sum(hist * np.log2(hist)))

    laplacian_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    f = np.fft.fft2(gray)
    fshift = np.fft.fftshift(f)
    magnitude = np.abs(fshift)
    h, w = magnitude.shape
    cy, cx = h // 2, w // 2
    radius = min(h, w) // 6
    y, x = np.ogrid[:h, :w]
    mask_low = (x - cx) ** 2 + (y - cy) ** 2 <= radius ** 2
    low_energy = magnitude[mask_low].sum()
    total_energy = magnitude.sum() + 1e-8
    high_freq_ratio = float(1.0 - (low_energy / total_energy))

    reasons = []
    live = True
    if lbp_entropy < 4.0:
        live = False
        reasons.append("texture too uniform (possible flat photo)")
    if laplacian_var < 15.0:
        live = False
        reasons.append("image too smooth/blurred (possible print or low-quality screen)")
    if high_freq_ratio > 0.55:
        live = False
        reasons.append("unusual high-frequency pattern detected (possible screen recapture)")

    score = round((lbp_entropy / 8.0) * 0.5 + min(laplacian_var / 100.0, 1.0) * 0.3 + (1 - high_freq_ratio) * 0.2, 3)

    return {
        "live": live,
        "score": score,
        "reason": "; ".join(reasons) if reasons else "Passed liveness checks"
    }



def process_video_for_enrollment(video_path: str) -> dict:
    """
    Extract frames from the enrolment video, detect faces, generate
    embeddings per frame, average them, and pick one representative
    face crop to use as the default profile picture.

    Returns:
      {
        "success": bool,
        "message": str,
        "embedding_json": str | None,   # JSON-encoded list of floats
        "profile_jpeg": bytes | None,   # representative face, JPEG-encoded
        "frames_used": int
      }
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return {"success": False, "message": "Could not open video file"}

    embeddings = []
    face_crops = []
    frame_count = 0
    sample_every = 15

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_count += 1
        if frame_count % sample_every != 0:
            continue
        face = detect_and_crop_face(frame)
        if face is None:
            continue
        face_resized = cv2.resize(face, (160, 160))
        try:
            emb = generate_embedding(face_resized)
            embeddings.append(emb)
            face_crops.append(face_resized)
        except Exception:
            continue

    cap.release()

    if len(embeddings) < 3:
        return {
            "success": False,
            "message": f"Only {len(embeddings)} valid face frames detected. Please record in better lighting and look directly at the camera.",
            "embedding_json": None,
            "profile_jpeg": None,
            "frames_used": len(embeddings)
        }

    avg_embedding = np.mean(embeddings, axis=0).tolist()
    embedding_json = json.dumps(avg_embedding)

    # Representative photo: the middle frame of the successful captures,
    # since early/late frames are more likely to catch blinking or motion.
    representative = face_crops[len(face_crops) // 2]
    success, buf = cv2.imencode(".jpg", representative)
    profile_jpeg = buf.tobytes() if success else None

    return {
        "success": True,
        "message": f"Face enrolled successfully from {len(embeddings)} frames",
        "embedding_json": embedding_json,
        "profile_jpeg": profile_jpeg,
        "frames_used": len(embeddings)
    }


def compare_face_to_stored(image_array: np.ndarray, stored_embedding_json: str) -> dict:
    """
    Compare a captured face to a stored embedding (loaded from the
    database as a JSON string, not a local file). Runs a liveness
    check FIRST — recognition never runs if liveness fails.
    """
    if not stored_embedding_json:
        return {"match": False, "live": None, "message": "No enrolled face found for this student"}

    face = detect_and_crop_face(image_array)
    if face is None:
        return {"match": False, "live": None, "message": "No face detected in submitted photo"}

    liveness = check_liveness(face)
    if not liveness["live"]:
        return {
            "match": False,
            "live": False,
            "liveness_score": liveness["score"],
            "message": f"Liveness check failed — please use your live face, not a photo or screen. ({liveness['reason']})"
        }

    face_resized = cv2.resize(face, (160, 160))
    try:
        live_emb = generate_embedding(face_resized)
    except Exception as e:
        return {"match": False, "live": True, "message": f"Embedding generation failed: {str(e)}"}

    stored_emb = json.loads(stored_embedding_json)

    live = np.array(live_emb)
    stored = np.array(stored_emb)
    similarity = float(np.dot(live, stored) / (np.linalg.norm(live) * np.linalg.norm(stored)))

    THRESHOLD = MATCH_THRESHOLD
    return {
        "match": similarity >= THRESHOLD,
        "live": True,
        "liveness_score": liveness["score"],
        "similarity": round(similarity, 4),
        "message": "Match found" if similarity >= THRESHOLD else "Face not recognised"
    }


def extract_face_jpeg_from_image(image_array: np.ndarray):
    """Used for manual profile picture uploads — detects and crops the
    face from an uploaded photo (not a video), returns JPEG bytes."""
    face = detect_and_crop_face(image_array)
    if face is None:
        return None
    success, buf = cv2.imencode(".jpg", face)
    return buf.tobytes() if success else None