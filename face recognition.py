import cv2
import numpy as np
import csv
import time
from facenet_pytorch import MTCNN, InceptionResnetV1
import torch
import os
from tkinter import Tk, filedialog

# Initialize MTCNN (face detector) and Inception Resnet V1 (for face recognition)
mtcnn = MTCNN(keep_all=True)
inception = InceptionResnetV1(pretrained='vggface2').eval()

# Set up time and attendance tracking
current_time = time.strftime("%Y-%m-%d %H-%M-%S")
detected_names = []
attendance_data = {}
# Set up time and attendance tracking (Modified to show time only as HH:MM:SS)
current_time1 = time.strftime("%H-%M-%S")  # Only hour, minute, and second


# Initialize the list to store face encodings
known_face_encodings = []  # Initialize it here to avoid the NameError

# Define how much time before resetting the system
#reset_interval = 3600
#start_time = time.time()

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
    face = Face(name, rrn, branch, image_path)
    face_data = face.face_upload()
    if face_data:
        encoding, name, rrn, branch = face_data
        # Append the face encoding and details to the global list
        known_face_encodings.append((encoding, name, rrn, branch))
        print(f"Uploaded face for {name}")
    else:
        print(f"Failed to upload face for {name}")


add_face('yaseen', 1170, 'AI&DS', 'yaseen.jpg')
add_face('naveed', 1152, 'AI&DS', 'naveed.jpg')
add_face('hameed', 1145, 'AI&DS', 'hameed.jpg')
add_face('vikinesh', 1146, 'AI&DS', 'viki.jpg')
add_face('chatu', 2381, 'AI&DS', 'chatu.jpg')
add_face('faaz', 4927, 'AI&DS', 'faaz.jpg')
add_face('hasim', 3852, 'AI&DS', 'hasim.jpg')
add_face('leo', 1743, 'AI&DS', 'leo.jpg')
add_face('maida', 5612, 'AI&DS', 'maida.jpg')
add_face('marofa', 8234, 'AI&DS', 'marofa.jpg')
add_face('nizam', 6723, 'AI&DS', 'nizam.jpg')
add_face('sabila', 3156, 'AI&DS', 'sabila.jpg')
add_face('sandy', 7812, 'AI&DS', 'sandy.jpg')
add_face('shabaz', 4590, 'AI&DS', 'shabaz.jpg')
add_face('shameer69', 6901, 'AI&DS', 'shameer69.jpg')
add_face('sheik_vili', 8420, 'AI&DS', 'sheik_vili.jpg')
add_face('stefina', 1204, 'AI&DS', 'stefina.jpg')
add_face('suthika', 2345, 'AI&DS', 'suthika.jpg')
add_face('swathy', 5678, 'AI&DS', 'swathy.jpg')
add_face('tawheed', 8765, 'AI&DS', 'tawheed.jpg')
add_face('vanathi', 4321, 'AI&DS', 'vanathi.jpg')
add_face('viswa', 3456, 'AI&DS', 'viswa.jpg')
add_face('zarah', 9876, 'AI&DS', 'zarah.jpg')


# --- New function to upload and process an image ---
def upload_and_recognize_image():
    # Hide the main Tkinter window
    root = Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title="Select an image", filetypes=[("Image files", "*.jpg;*.jpeg;*.png")])
    if not file_path:
        print("No file selected.")
        return

    img = cv2.imread(file_path)
    if img is None:
        print(f"Image not found: {file_path}")
        return

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    boxes, probs = mtcnn.detect(rgb_img)
    faces = mtcnn(rgb_img)

    if boxes is not None and faces is not None:
        for i, face in enumerate(faces):
            if face is not None:
                face_embedding = inception(face.unsqueeze(0))
                distances = [np.linalg.norm(face_embedding.detach().numpy() - encoding) for encoding, _, _, _ in known_face_encodings]
                min_distance_index = np.argmin(distances)
                name = "unknown"
                if distances[min_distance_index] < 0.8:
                    name = known_face_encodings[min_distance_index][1]
                x_min, y_min, x_max, y_max = boxes[i].tolist()
                cv2.rectangle(img, (int(x_min), int(y_min)), (int(x_max), int(y_max)), (0, 0, 255), 2)
                cv2.putText(img, name, (int(x_min), int(y_min) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        # Save the processed image
        base_name = os.path.basename(file_path)
        save_path = os.path.join(os.path.dirname(file_path), f"recognized_{base_name}")
        cv2.imwrite(save_path, img)
        print(f"Processed image saved as {save_path}")
        cv2.imshow("Recognized Image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("No faces detected in the image.")

# --- Uncomment below to use webcam as before ---
# video_capture = cv2.VideoCapture(0)
# while True:
#     ...existing code...
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break
# video_capture.release()
# cv2.destroyAllWindows()

# --- Call the new function to upload and process an image ---
if __name__ == "__main__":
    upload_and_recognize_image()

