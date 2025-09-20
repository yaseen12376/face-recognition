from fastapi import FastAPI, Request, Form, File, UploadFile, HTTPException, Depends, status
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
import cv2
import numpy as np
import csv
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
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

# Security
SECRET_KEY = "your-secret-key-here"  # Change this in production
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 30

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
security = HTTPBearer()

# Templates and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Initialize MTCNN (face detector) and Inception Resnet V1 (for face recognition)
mtcnn = MTCNN(keep_all=True)
inception = InceptionResnetV1(pretrained='vggface2').eval()

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
            # Load the image and extract face embeddings
            img = cv2.imread(self.image)
            if img is None:
                print(f"Image not found: {self.image}")
                return False
            
            # Convert BGR to RGB (OpenCV loads images in BGR format)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            faces = mtcnn(img)  # Detect faces in the image
            if faces is not None:
                for face in faces:
                    # Ensure the image is in the right format for InceptionResnetV1
                    face_embedding = inception(face.unsqueeze(0))  # Add batch dimension
                    # Return face encoding and the person's details (name, rrn, branch)
                    return face_embedding.detach().numpy(), self.name, self.rrn, self.branch
            return None
        except Exception as e:
            print(f"Error loading image: {e}")
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

def recognize_face_from_image(image_array):
    """Recognize faces in the given image array"""
    try:
        # Convert to RGB if it's BGR
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            rgb_img = cv2.cvtColor(image_array, cv2.COLOR_BGR2RGB)
        else:
            rgb_img = image_array
            
        # Detect faces
        boxes, probs = mtcnn.detect(rgb_img)
        faces = mtcnn(rgb_img)
        
        results = []
        
        if boxes is not None and faces is not None:
            for i, face in enumerate(faces):
                if face is not None:
                    face_embedding = inception(face.unsqueeze(0))
                    distances = [np.linalg.norm(face_embedding.detach().numpy() - encoding) 
                               for encoding, _, _, _ in known_face_encodings]
                    
                    if distances:
                        min_distance_index = np.argmin(distances)
                        if distances[min_distance_index] < 0.9:
                            _, name, rrn, branch = known_face_encodings[min_distance_index]
                            results.append({
                                "name": name,
                                "rrn": rrn,
                                "branch": branch,
                                "confidence": 1 - distances[min_distance_index],
                                "box": boxes[i].tolist()
                            })
                        else:
                            results.append({
                                "name": "unknown",
                                "rrn": None,
                                "branch": None,
                                "confidence": 0,
                                "box": boxes[i].tolist()
                            })
        
        return results
    except Exception as e:
        print(f"Error in face recognition: {e}")
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
            if result["name"] != "unknown":
                attendance_record = {
                    "name": result["name"],
                    "rrn": result["rrn"],
                    "branch": result["branch"],
                    "time": current_time,
                    "date": current_date,
                    "status": "Present"
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