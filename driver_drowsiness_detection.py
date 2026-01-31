import cv2
import numpy as np
import mediapipe as mp
import tkinter as tk
from PIL import Image, ImageTk
import sys

# ==========================
# MediaPipe Initialization
# ==========================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True
)

# ==========================
# Thresholds & Counters
# ==========================
EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 0.4
CONSEC_FRAMES = 5

frame_count = 0
yawn_frame_count = 0
eye_closed_count = 0
yawn_times = 0

# ==========================
# Landmark Indices
# ==========================
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14]

# ==========================
# Utility Functions
# ==========================
def calculate_EAR(landmarks, indices, w, h):
    eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    hor = np.linalg.norm(eye[0] - eye[3])
    ver = np.linalg.norm(eye[1] - eye[5]) + np.linalg.norm(eye[2] - eye[4])
    return ver / (2.0 * hor)

def calculate_MAR(landmarks, indices, w, h):
    mouth = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
    hor = np.linalg.norm(mouth[0] - mouth[1])
    ver = np.linalg.norm(mouth[2] - mouth[3])
    return ver / hor

# ==========================
# Tkinter UI Setup
# ==========================
window = tk.Tk()
window.title("Driver Monitoring System")
window.geometry("900x720")
window.configure(bg="#0f172a")

# Main Container
main = tk.Frame(window, bg="#0f172a")
main.pack(fill="both", expand=True, padx=20, pady=20)

# Title
title = tk.Label(
    main,
    text="Driver Drowsiness & Yawning Detection",
    font=("Segoe UI", 20, "bold"),
    bg="#0f172a",
    fg="white"
)
title.pack(pady=10)

# Status Card
status_card = tk.Frame(main, bg="#1e293b")
status_card.pack(fill="x", pady=10)

status_label = tk.Label(
    status_card,
    text="Status: Waiting...",
    font=("Segoe UI", 14),
    bg="#1e293b",
    fg="#38bdf8"
)
status_label.pack(pady=10)

# Counter Card
counter_card = tk.Frame(main, bg="#1e293b")
counter_card.pack(fill="x", pady=10)

eye_label = tk.Label(
    counter_card,
    text="Eye Closed Count: 0",
    font=("Segoe UI", 12),
    bg="#1e293b",
    fg="white"
)
eye_label.pack(pady=5)

yawn_label = tk.Label(
    counter_card,
    text="Yawning Count: 0",
    font=("Segoe UI", 12),
    bg="#1e293b",
    fg="white"
)
yawn_label.pack(pady=5)

# Video Area
video_frame = tk.Frame(main, bg="#020617", height=400)
video_frame.pack(fill="both", expand=True, pady=15)

video_label = tk.Label(video_frame, bg="#020617")
video_label.pack(expand=True)

# Buttons
button_frame = tk.Frame(main, bg="#0f172a")
button_frame.pack(pady=15)

is_live = False

def toggle_live():
    global is_live
    is_live = not is_live
    start_btn.config(text="Stop Live" if is_live else "Start Live")

def exit_app():
    cap.release()
    cv2.destroyAllWindows()
    window.destroy()

start_btn = tk.Button(
    button_frame,
    text="Start Live",
    font=("Segoe UI", 12, "bold"),
    bg="#38bdf8",
    fg="black",
    width=15,
    relief="flat",
    command=toggle_live
)
start_btn.grid(row=0, column=0, padx=10)

exit_btn = tk.Button(
    button_frame,
    text="Exit",
    font=("Segoe UI", 12, "bold"),
    bg="#ef4444",
    fg="white",
    width=15,
    relief="flat",
    command=exit_app
)
exit_btn.grid(row=0, column=1, padx=10)

# ==========================
# Camera Setup
# ==========================
cap = cv2.VideoCapture(0)

# ==========================
# Main Loop
# ==========================
def update():
    global frame_count, yawn_frame_count, eye_closed_count, yawn_times

    if is_live:
        success, frame = cap.read()
        if not success:
            return

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        h, w, _ = frame.shape

        status = "Active"

        if results.multi_face_landmarks:
            for face in results.multi_face_landmarks:
                EAR = (
                    calculate_EAR(face.landmark, LEFT_EYE, w, h) +
                    calculate_EAR(face.landmark, RIGHT_EYE, w, h)
                ) / 2

                MAR = calculate_MAR(face.landmark, MOUTH, w, h)

                if EAR < EAR_THRESHOLD:
                    frame_count += 1
                    if frame_count >= CONSEC_FRAMES:
                        eye_closed_count += 1
                        eye_label.config(text=f"Eye Closed Count: {eye_closed_count}")
                        status = "DROWSY - Eyes Closed"
                else:
                    frame_count = 0

                if MAR > MAR_THRESHOLD:
                    yawn_frame_count += 1
                    if yawn_frame_count >= 3:
                        yawn_times += 1
                        yawn_label.config(text=f"Yawning Count: {yawn_times}")
                        status = "DROWSY - Yawning"
                else:
                    yawn_frame_count = 0

                for idx in LEFT_EYE + RIGHT_EYE + MOUTH:
                    x = int(face.landmark[idx].x * w)
                    y = int(face.landmark[idx].y * h)
                    cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)

        else:
            status = "No Face Detected"

        status_label.config(
            text=f"Status: {status}",
            fg="#22c55e" if status == "Active" else "#ef4444"
        )

        img = ImageTk.PhotoImage(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
        video_label.imgtk = img
        video_label.config(image=img)

    window.after(10, update)

update()
window.mainloop()

cap.release()
cv2.destroyAllWindows()
