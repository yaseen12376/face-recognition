from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pathlib import Path
from typing import Optional, List

import base64
import csv
import json
from datetime import datetime, timedelta

import cv2
import numpy as np
from insightface.app import FaceAnalysis
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel, Field

# Initialize FastAPI app
app = FastAPI(title="Smart Attendance System", description="Face Recognition based Attendance System")

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
STUDENTS_FILE = DATA_DIR / "students.json"
ATTENDANCE_FILE = DATA_DIR / "attendance.csv"

# Image quality configuration for RetinaFace + ArcFace
class ImageConfig:
    # Face detection thresholds - RetinaFace works better with different thresholds
    MIN_FACE_SIZE = 30  # Minimum face size in pixels
    RETINAFACE_THRESHOLD = 0.7  # RetinaFace detection threshold
    ARCFACE_SIMILARITY_THRESHOLD = 0.4  # ArcFace cosine similarity threshold
    
    # Image enhancement settings
    SHARPNESS_FACTOR = 1.5  # Reduced from 2.0 as ArcFace is more robust
    CONTRAST_FACTOR = 1.2   # Reduced from 1.3
    BRIGHTNESS_FACTOR = 1.1 # Slight brightness boost
    
    # Advanced processing settings
    NOISE_REDUCTION = True
    HISTOGRAM_EQUALIZATION = True
    ADAPTIVE_THRESHOLD = True
    FACE_ALIGNMENT = True  # Enable proper face alignment
    STANDARD_FACE_SIZE = (112, 112)  # ArcFace standard input size


DEFAULT_STUDENTS = [
    {"name": "yaseen", "rrn": 1170, "branch": "AI&DS", "image": "yaseen.jpg"},
    {"name": "naveed", "rrn": 1152, "branch": "AI&DS", "image": "naveed.jpg"},
    {"name": "hameed", "rrn": 1145, "branch": "AI&DS", "image": "hameed.jpg"},
    {"name": "vikinesh", "rrn": 1146, "branch": "AI&DS", "image": "viki.jpg"},
    {"name": "chatu", "rrn": 2381, "branch": "AI&DS", "image": "chatu.jpg"},
    {"name": "faaz", "rrn": 4927, "branch": "AI&DS", "image": "faaz.jpg"},
    {"name": "hasim", "rrn": 3852, "branch": "AI&DS", "image": "hasim.jpg"},
    {"name": "leo", "rrn": 1743, "branch": "AI&DS", "image": "leo.jpg"},
    {"name": "maida", "rrn": 5612, "branch": "AI&DS", "image": "maida.jpg"},
    {"name": "marofa", "rrn": 8234, "branch": "AI&DS", "image": "marofa.jpg"},
    {"name": "nizam", "rrn": 6723, "branch": "AI&DS", "image": "nizam.jpg"},
    {"name": "sabila", "rrn": 3156, "branch": "AI&DS", "image": "sabila.jpg"},
    {"name": "sandy", "rrn": 7812, "branch": "AI&DS", "image": "sandy.jpg"},
    {"name": "shabaz", "rrn": 4590, "branch": "AI&DS", "image": "shabaz.jpg"},
    {"name": "shameer69", "rrn": 6901, "branch": "AI&DS", "image": "shameer69.jpg"},
    {"name": "sheik_vili", "rrn": 8420, "branch": "AI&DS", "image": "sheik_vili.jpg"},
    {"name": "stefina", "rrn": 1204, "branch": "AI&DS", "image": "stefina.jpg"},
    {"name": "suthika", "rrn": 2345, "branch": "AI&DS", "image": "suthika.jpg"},
    {"name": "swathy", "rrn": 5678, "branch": "AI&DS", "image": "swathy.jpg"},
    {"name": "tawheed", "rrn": 8765, "branch": "AI&DS", "image": "tawheed.jpg"},
    {"name": "vanathi", "rrn": 4321, "branch": "AI&DS", "image": "vanathi.jpg"},
    {"name": "viswa", "rrn": 3456, "branch": "AI&DS", "image": "viswa.jpg"},
    {"name": "zarah", "rrn": 9876, "branch": "AI&DS", "image": "zarah.jpg"}
]


class AttendanceRecord(BaseModel):
    name: str
    rrn: int
    branch: str
    time: str
    date: str
    status: str = "Present"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    detection_confidence: Optional[float] = Field(default=None, ge=0.0, le=1.0)

    @property
    def unique_key(self) -> str:
        return f"{self.name}_{self.date}"


class AttendanceStore:
    def __init__(self, csv_path: Path):
        self.csv_path = csv_path
        self.records: dict[str, AttendanceRecord] = {}
        self._load_existing()

    def _load_existing(self) -> None:
        if not self.csv_path.exists():
            return

        try:
            with self.csv_path.open(newline="", mode="r", encoding="utf-8") as csvfile:
                reader = csv.DictReader(csvfile)
                for row in reader:
                    try:
                        record = AttendanceRecord(**row)
                        self.records[record.unique_key] = record
                    except Exception as exc:
                        print(f"Skipping invalid attendance row {row}: {exc}")
        except Exception as exc:
            print(f"Error loading attendance history: {exc}")

    def _append_csv(self, record: AttendanceRecord) -> None:
        file_exists = self.csv_path.exists()
        with self.csv_path.open(mode="a", newline="", encoding="utf-8") as csvfile:
            fieldnames = list(record.dict().keys())
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            if not file_exists:
                writer.writeheader()
            writer.writerow(record.dict())

    def add(self, record: AttendanceRecord) -> bool:
        if record.unique_key in self.records:
            return False

        self.records[record.unique_key] = record
        self._append_csv(record)
        return True

    def list_records(self) -> List[dict]:
        return [record.dict() for record in self.records.values()]


def load_student_roster() -> List[dict]:
    if STUDENTS_FILE.exists():
        try:
            with STUDENTS_FILE.open("r", encoding="utf-8") as fp:
                data = json.load(fp)
                if isinstance(data, list):
                    return data
        except Exception as exc:
            print(f"Error reading {STUDENTS_FILE}: {exc}")
    else:
        try:
            with STUDENTS_FILE.open("w", encoding="utf-8") as fp:
                json.dump(DEFAULT_STUDENTS, fp, indent=2)
                return DEFAULT_STUDENTS
        except Exception as exc:
            print(f"Unable to create {STUDENTS_FILE}: {exc}")
    return DEFAULT_STUDENTS

# Security
SECRET_KEY = "your-secret-key-here"  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize RetinaFace and ArcFace models
print("Initializing RetinaFace and ArcFace models...")
try:
    # Initialize Face Analysis app with ArcFace model (MobileFaceNet backbone)
    face_app = FaceAnalysis(providers=['CPUExecutionProvider'])
    face_app.prepare(ctx_id=0, det_size=(640, 640))
    print("✓ ArcFace model initialized successfully")
except Exception as e:
    print(f"Error initializing ArcFace: {e}")
    face_app = None

# Global variables for face recognition
known_face_encodings = []
attendance_store = AttendanceStore(ATTENDANCE_FILE)

class Face:
    def __init__(self, name, rrn, branch, image):
        self.name = name
        self.rrn = rrn
        self.image = image
        self.branch = branch

    def display_face(self):
        return f"{self.name},{self.rrn},{self.image}"

    def face_upload(self):
        try:
            # Load the image using OpenCV
            img = cv2.imread(self.image)
            if img is None:
                print(f"Image not found: {self.image}")
                return False
            
            # Convert BGR to RGB for RetinaFace
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Use ArcFace (InsightFace) for face detection and embedding
            if face_app is None:
                print("Face analysis app not initialized")
                return False
                
            faces = face_app.get(img_rgb)
            
            if faces and len(faces) > 0:
                # Get the first (most confident) face
                face = faces[0]
                face_embedding = face.embedding  # ArcFace embedding
                
                # Return face encoding and the person's details
                return face_embedding, self.name, self.rrn, self.branch
            else:
                print(f"No faces detected in {self.image}")
                return None
                
        except Exception as e:
            print(f"Error loading image {self.image}: {e}")
            return None

def add_face(name, rrn, branch, image_path):
    """Add a face to the known faces database"""
    face = Face(name, rrn, branch, image_path)
    face_data = face.face_upload()
    if face_data:
        encoding, name, rrn, branch = face_data
        # Append the face encoding and details to the global list
        known_face_encodings.append((encoding, name, rrn, branch))
        print(f"Uploaded face for {name}")
        return True
    else:
        print(f"Failed to upload face for {name}")
        return False

def initialize_known_faces():
    """Initialize all known faces from the image files"""
    known_face_encodings.clear()
    student_roster = load_student_roster()

    for student in student_roster:
        try:
            add_face(student["name"], student["rrn"], student["branch"], student["image"])
        except KeyError as exc:
            print(f"Student entry missing field {exc}: {student}")

# Dummy user database (replace with real database in production)
fake_users_db = {
    "professor@college.edu": {
        "username": "professor@college.edu",
        "full_name": "Professor Anya",
        "hashed_password": pwd_context.hash("password123"),
        "disabled": False,
    }
}

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(db, username: str):
    if username in db:
        user_dict = db[username]
        return user_dict

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
    if not user:
        return False
    if not verify_password(password, user["hashed_password"]):
        return False
    return user

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow() + timedelta(minutes=15)
    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)
    return encoded_jwt

async def get_current_user(credentials: HTTPAuthorizationCredentials = Depends(security)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(credentials.credentials, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = get_user(fake_users_db, username=username)
    if user is None:
        raise credentials_exception
    return user

def detect_and_align_faces(image_array):
    """Detect and align faces using InsightFace (which includes RetinaFace detection)"""
    try:
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            rgb_img = image_array
        else:
            rgb_img = image_array
            
        # Use InsightFace for detection and alignment (includes RetinaFace detector)
        if face_app is None:
            print("Face analysis app not initialized")
            return []
            
        faces = face_app.get(rgb_img)
        
        faces_info = []
        if faces:
            for face in faces:
                # Convert bbox format from InsightFace
                bbox = face.bbox.astype(int)  # [x1, y1, x2, y2]
                confidence = face.det_score
                landmarks = face.kps if hasattr(face, 'kps') else None
                
                # Extract aligned face (InsightFace already provides aligned faces)
                aligned_face = rgb_img[bbox[1]:bbox[3], bbox[0]:bbox[2]]
                
                # Resize to standard size for consistent processing
                if aligned_face.shape[0] > 0 and aligned_face.shape[1] > 0:
                    aligned_face = cv2.resize(aligned_face, ImageConfig.STANDARD_FACE_SIZE)
                
                faces_info.append({
                    'aligned_face': aligned_face,
                    'bbox': bbox.tolist(),
                    'confidence': float(confidence),
                    'landmarks': landmarks.tolist() if landmarks is not None else None,
                    'embedding': face.embedding  # Pre-computed ArcFace embedding
                })
        
        return faces_info
    except Exception as e:
        print(f"Error in InsightFace detection: {e}")
        return []

def align_face_with_landmarks(image, landmarks, bbox):
    """Align face using facial landmarks"""
    try:
        # Simple face extraction for now - can be enhanced with proper alignment
        x1, y1, x2, y2 = bbox
        face_img = image[y1:y2, x1:x2]
        
        # Resize to standard size for consistent processing
        if face_img.shape[0] > 0 and face_img.shape[1] > 0:
            face_img = cv2.resize(face_img, ImageConfig.STANDARD_FACE_SIZE)  # Standard size for ArcFace
        
        return face_img
    except Exception as e:
        print(f"Error in face alignment: {e}")
        return None

def get_arcface_embedding(face_image):
    """Generate ArcFace embedding for a face image"""
    try:
        if face_app is None:
            print("Face analysis app not initialized")
            return None
            
        # Use InsightFace to get embedding
        faces = face_app.get(face_image)
        if faces and len(faces) > 0:
            return faces[0].embedding
        else:
            print("No face detected in the provided image")
            return None
    except Exception as e:
        print(f"Error generating ArcFace embedding: {e}")
        return None

def recognize_face_from_image(image_array):
    """Advanced face recognition using RetinaFace detection and ArcFace embeddings"""
    try:
        # Convert to RGB if needed
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            rgb_img = image_array
        else:
            rgb_img = image_array
            
        # Use RetinaFace for face detection and alignment
        faces_info = detect_and_align_faces(rgb_img)
        
        results = []
        
        if faces_info:
            for i, face_info in enumerate(faces_info):
                try:
                    # Get ArcFace embedding
                    if 'embedding' in face_info:
                        # Use pre-computed embedding from InsightFace fallback
                        face_embedding = face_info['embedding']
                    else:
                        # Generate embedding using ArcFace
                        aligned_face = face_info['aligned_face']
                        if aligned_face is not None:
                            face_embedding = get_arcface_embedding(aligned_face)
                        else:
                            continue
                    
                    if face_embedding is not None:
                        # Calculate cosine similarities with known faces (better for ArcFace)
                        similarities = []
                        for encoding, _, _, _ in known_face_encodings:
                            # Cosine similarity for ArcFace embeddings
                            similarity = np.dot(face_embedding, encoding) / (
                                np.linalg.norm(face_embedding) * np.linalg.norm(encoding)
                            )
                            similarities.append(similarity)
                        
                        if similarities:
                            max_similarity_index = np.argmax(similarities)
                            max_similarity = similarities[max_similarity_index]
                            
                            # ArcFace uses cosine similarity, higher is better (threshold ~0.3-0.6)
                            if max_similarity > ImageConfig.ARCFACE_SIMILARITY_THRESHOLD:  # Threshold for ArcFace recognition
                                _, name, rrn, branch = known_face_encodings[max_similarity_index]
                                confidence = max_similarity  # Similarity as confidence
                                
                                results.append({
                                    "name": name,
                                    "rrn": rrn,
                                    "branch": branch,
                                    "confidence": float(confidence),
                                    "box": face_info['bbox'],
                                    "detection_confidence": float(face_info['confidence']),
                                    "similarity": float(max_similarity),
                                    "quality_assessment": "good" if face_info['confidence'] > 0.8 else "acceptable"
                                })
                            else:
                                results.append({
                                    "name": "unknown",
                                    "rrn": None,
                                    "branch": None,
                                    "confidence": 0,
                                    "box": face_info['bbox'],
                                    "detection_confidence": float(face_info['confidence']),
                                    "similarity": float(max_similarity),
                                    "quality_assessment": "unknown_person"
                                })
                except Exception as face_err:
                    print(f"Error processing face {i}: {face_err}")
                    continue
        
        return results
    except Exception as e:
        print(f"Error in RetinaFace + ArcFace recognition: {e}")
        return []

# Initialize known faces when app starts
initialize_known_faces()

# Routes
@app.get("/", response_class=HTMLResponse)
async def login_page(request: Request):
    """Serve the login page"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.get("/login", response_class=HTMLResponse)
async def login_page_get(request: Request):
    """Serve the login page"""
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login")
async def login_for_access_token(request: Request, email: str = Form(...), password: str = Form(...)):
    """Handle login form submission"""
    user = authenticate_user(fake_users_db, email, password)
    if not user:
        return templates.TemplateResponse("login.html", {
            "request": request, 
            "error": "Incorrect email or password"
        })
    access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(
        data={"sub": user["username"]}, expires_delta=access_token_expires
    )
    response = RedirectResponse(url="/dashboard", status_code=status.HTTP_302_FOUND)
    response.set_cookie(key="access_token", value=f"Bearer {access_token}", httponly=True)
    return response

@app.get("/dashboard", response_class=HTMLResponse)
async def dashboard(request: Request):
    """Serve the dashboard page"""
    # Check if user is authenticated via cookie
    token = request.cookies.get("access_token")
    if not token:
        return RedirectResponse(url="/login")
    
    try:
        # Remove "Bearer " prefix if present
        if token.startswith("Bearer "):
            token = token[7:]
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            return RedirectResponse(url="/login")
    except JWTError:
        return RedirectResponse(url="/login")
    
    user = get_user(fake_users_db, username=username)
    if user is None:
        return RedirectResponse(url="/login")
    
    current_date = datetime.now().strftime("%B %d, %Y")
    return templates.TemplateResponse("dashboard.html", {
        "request": request,
        "user": user,
        "current_date": current_date
    })

@app.post("/logout")
async def logout():
    """Handle logout"""
    response = RedirectResponse(url="/login", status_code=status.HTTP_302_FOUND)
    response.delete_cookie("access_token")
    return response

@app.post("/recognize")
async def recognize_face(request: Request, image: UploadFile = File(...)):
    """Handle face recognition from uploaded image"""
    try:
        # Read image file
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Recognize faces
        results = recognize_face_from_image(img)
        
        # Mark attendance for recognized faces
        attendance_records = []
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        for result in results:
            if result["name"] != "unknown":
                record_model = AttendanceRecord(
                    name=result["name"],
                    rrn=result["rrn"],
                    branch=result["branch"],
                    time=current_time,
                    date=current_date,
                    status="Present",
                    confidence=float(result.get("confidence", 0.0)),
                    detection_confidence=float(result.get("detection_confidence", 0.0)) if result.get("detection_confidence") is not None else None,
                )
                was_added = attendance_store.add(record_model)
                attendance_records.append({
                    **record_model.dict(),
                    "already_recorded": not was_added
                })
        
        return JSONResponse({
            "success": True,
            "results": results,
            "attendance_records": attendance_records,
            "message": f"Recognized {len(attendance_records)} students"
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

@app.post("/capture-attendance")
async def capture_attendance(request: Request, image_data: str = Form(...)):
    """Handle attendance capture from webcam"""
    try:
        # Decode base64 image
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        
        # Convert to numpy array
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image data")
        
        # Recognize faces
        results = recognize_face_from_image(img)
        
        # Mark attendance
        attendance_records = []
        current_time = datetime.now().strftime("%H:%M:%S")
        current_date = datetime.now().strftime("%Y-%m-%d")
        
        for result in results:
            if result["name"] not in ["unknown", "low_quality_detection", "low_confidence"]:
                record_model = AttendanceRecord(
                    name=result["name"],
                    rrn=result["rrn"],
                    branch=result["branch"],
                    time=current_time,
                    date=current_date,
                    status="Present",
                    confidence=float(result.get("confidence", 0.8)),
                    detection_confidence=float(result.get("detection_confidence", 0.8)) if result.get("detection_confidence") is not None else None,
                )
                was_added = attendance_store.add(record_model)
                attendance_records.append({
                    **record_model.dict(),
                    "already_recorded": not was_added
                })
        
        return JSONResponse({
            "success": True,
            "results": results,
            "attendance_records": attendance_records
        })
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing capture: {str(e)}")

@app.get("/attendance")
async def get_attendance(request: Request):
    """Get attendance records"""
    token = request.cookies.get("access_token")
    if not token:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    return JSONResponse({
        "success": True,
        "attendance": attendance_store.list_records()
    })

@app.get("/students")
async def get_students():
    """Get list of all registered students"""
    students = []
    for encoding, name, rrn, branch in known_face_encodings:
        students.append({
            "name": name,
            "rrn": rrn,
            "branch": branch
        })
    return JSONResponse({
        "success": True,
        "students": students
    })

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="127.0.0.1", port=8000, reload=True)