# Smart Attendance System

This FastAPI project captures classroom attendance using RetinaFace detection and ArcFace recognition.

## Configure the student roster

Student identities are now stored in `data/students.json`. Update this file to add, edit, or remove entries without touching the application code:

```json
[
	{ "name": "jane", "rrn": 1010, "branch": "AI&DS", "image": "jane.jpg" }
]
```

Make sure the `image` value matches a photo placed in the project root.

## Attendance history

Each time a student is recognized, the system records the entry in `data/attendance.csv`. Duplicate detections on the same day are ignored automatically, so the CSV always reflects unique daily attendance.

## Run locally

```powershell
cd face-recognition
uvicorn app:app --reload
```

Open `http://127.0.0.1:8000` in your browser to access the dashboard.