from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import cv2
import numpy as np
import csv
import time
import insightface
from insightface.app import FaceAnalysis
import torch
import os
import io
from datetime import datetime, timedelta
from typing import Optional, List
import base64
from PIL import Image
import json
from passlib.context import CryptContext
from jose import JWTError, jwt

# Initialize FastAPI app
app = FastAPI(title="Smart Attendance System", description="Face Recognition based Attendance System")

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
attendance_data = {}
detected_names = []

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
    face_data = [
        ('yaseen', 1170, 'AI&DS', 'yaseen.jpg'),
        ('naveed', 1152, 'AI&DS', 'naveed.jpg'),
        ('hameed', 1145, 'AI&DS', 'hameed.jpg'),
        ('vikinesh', 1146, 'AI&DS', 'viki.jpg'),
        ('chatu', 2381, 'AI&DS', 'chatu.jpg'),
        ('faaz', 4927, 'AI&DS', 'faaz.jpg'),
        ('hasim', 3852, 'AI&DS', 'hasim.jpg'),
        ('leo', 1743, 'AI&DS', 'leo.jpg'),
        ('maida', 5612, 'AI&DS', 'maida.jpg'),
        ('marofa', 8234, 'AI&DS', 'marofa.jpg'),
        ('nizam', 6723, 'AI&DS', 'nizam.jpg'),
        ('sabila', 3156, 'AI&DS', 'sabila.jpg'),
        ('sandy', 7812, 'AI&DS', 'sandy.jpg'),
        ('shabaz', 4590, 'AI&DS', 'shabaz.jpg'),
        ('shameer69', 6901, 'AI&DS', 'shameer69.jpg'),
        ('sheik_vili', 8420, 'AI&DS', 'sheik_vili.jpg'),
        ('stefina', 1204, 'AI&DS', 'stefina.jpg'),
        ('suthika', 2345, 'AI&DS', 'suthika.jpg'),
        ('swathy', 5678, 'AI&DS', 'swathy.jpg'),
        ('tawheed', 8765, 'AI&DS', 'tawheed.jpg'),
        ('vanathi', 4321, 'AI&DS', 'vanathi.jpg'),
        ('viswa', 3456, 'AI&DS', 'viswa.jpg'),
        ('zarah', 9876, 'AI&DS', 'zarah.jpg')
    ]
    
    for name, rrn, branch, image_path in face_data:
        add_face(name, rrn, branch, image_path)

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
                attendance_record = {
                    "name": result["name"],
                    "rrn": result["rrn"],
                    "branch": result["branch"],
                    "time": current_time,
                    "date": current_date,
                    "status": "Present",
                    "confidence": result["confidence"]
                }
                attendance_records.append(attendance_record)
                
                # Store in global attendance data
                key = f"{result['name']}_{current_date}"
                if key not in attendance_data:
                    attendance_data[key] = attendance_record
        
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
                attendance_record = {
                    "name": result["name"],
                    "rrn": result["rrn"],
                    "branch": result["branch"],
                    "time": current_time,
                    "date": current_date,
                    "status": "Present",
                    "confidence": result.get("confidence", 0.8),  # Include confidence score
                    "detection_confidence": result.get("detection_confidence", 0.8)
                }
                attendance_records.append(attendance_record)
                
                # Store in global attendance data
                key = f"{result['name']}_{current_date}"
                if key not in attendance_data:
                    attendance_data[key] = attendance_record
        
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
        "attendance": list(attendance_data.values())
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