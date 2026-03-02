import cv2
import time
import os
import threading
from datetime import datetime
from tkinter import Tk, Label, StringVar, Button, Toplevel, Listbox, Scrollbar, END
from PIL import Image, ImageTk
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
from suspicious_logic import SuspiciousDetector
import pandas as pd
import numpy as np
from collections import deque
np.float = float

# === CONFIG ===
VIDEO_SOURCE = 'videos/sample.mp4'
OUTPUT_FOLDER = 'suspicious_clips'
CLIP_DURATION = 5
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

# === YOLO + Tracker ===
model = YOLO("yolov8m.pt")  # medium accuracy
tracker = DeepSort(max_age=30, n_init=5, nn_budget=100, max_cosine_distance=0.3)
detector = SuspiciousDetector()

# === GUI Globals ===
gui_running = True
id_var = None
root = None
suspicious_ids = set()
recording_dict = {}
logged_status = {}
history = {}

# === Log Setup ===
log_path = "suspicious_log.txt"
if not os.path.exists(log_path):
    with open(log_path, "w") as f:
        f.write("Suspicious Behavior Log\n")
        f.write("Time\t\t\t\tTrack ID\tStatus\n")
        f.write("--------------------------------------------------\n")

def write_log(track_id, status):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(log_path, "a") as f:
        f.write(f"{timestamp}\t{track_id}\t\t{status.capitalize()}\n")

def save_report():
    df = pd.read_csv(log_path, sep="\t", skiprows=3, names=["Time", "Track ID", "Status"])
    df.to_csv("suspicious_report.csv", index=False)

def open_clip_viewer():
    top = Toplevel(root)
    top.title("Saved Suspicious Clips")
    top.geometry("400x300")
    scrollbar = Scrollbar(top)
    scrollbar.pack(side="right", fill="y")
    listbox = Listbox(top, yscrollcommand=scrollbar.set, font=("Arial", 12))
    clips = [f for f in os.listdir(OUTPUT_FOLDER) if f.endswith(".mp4")]
    for clip in clips:
        listbox.insert(END, clip)
    listbox.pack(side="left", fill="both", expand=True)
    scrollbar.config(command=listbox.yview)
    def play_clip(event):
        idx = listbox.curselection()
        if idx:
            clip_path = os.path.join(OUTPUT_FOLDER, listbox.get(idx[0]))
            os.system(f'start "" "{clip_path}"')
    listbox.bind("<Double-Button-1>", play_clip)

def run_gui():
    global id_var, root
    root = Tk()
    root.title("Suspicious Activity Dashboard")
    root.geometry("400x200")
    id_var = StringVar()
    id_var.set("Suspicious IDs: 0")

    Label(root, text="Real-Time Suspicious Detector", font=("Helvetica", 16)).pack(pady=10)
    Label(root, textvariable=id_var, font=("Helvetica", 14)).pack()

    Button(root, text="📂 View Saved Clips", command=open_clip_viewer, width=25).pack(pady=10)
    Button(root, text="📄 Generate Report", command=save_report, width=25).pack()

    def on_close():
        global gui_running
        gui_running = False
        root.destroy()

    root.protocol("WM_DELETE_WINDOW", on_close)
    root.mainloop()

def run_detection():
    global gui_running
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    width = int(cap.get(3))
    height = int(cap.get(4))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')

    while cap.isOpened() and gui_running:
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame, conf=0.5)[0]
        detections = []

        for r in results.boxes.data.tolist():
            x1, y1, x2, y2, score, cls = r
            if int(cls) == 0:
                detections.append(([x1, y1, x2 - x1, y2 - y1], score, "person"))

        tracks = tracker.update_tracks(detections, frame=frame)

        for track in tracks:
            if not track.is_confirmed():
                continue
            track_id = track.track_id
            l, t, r, b = map(int, track.to_ltrb())
            cx, cy = (l + r) // 2, (t + b) // 2

            # Smooth path
            history.setdefault(track_id, deque(maxlen=5)).append((cx, cy))
            smooth_x = int(sum(p[0] for p in history[track_id]) / len(history[track_id]))
            smooth_y = int(sum(p[1] for p in history[track_id]) / len(history[track_id]))

            detector.update(track_id, (smooth_x, smooth_y))
            status = detector.check(track_id)

            if status and logged_status.get(track_id) != status:
                write_log(track_id, status)
                logged_status[track_id] = status
                suspicious_ids.add(track_id)
                filename = f"{OUTPUT_FOLDER}/suspicious_{track_id}_{int(time.time())}.mp4"
                out = cv2.VideoWriter(filename, fourcc, fps, (width, height))
                recording_dict[track_id] = {"writer": out, "start": time.time()}

            if track_id in recording_dict:
                writer = recording_dict[track_id]
                writer["writer"].write(frame)
                if time.time() - writer["start"] > CLIP_DURATION:
                    writer["writer"].release()
                    del recording_dict[track_id]

            color = (0, 0, 255) if status else (0, 255, 0)
            label = f"ID {track_id}" + (f" - {status}" if status else "")
            cv2.rectangle(frame, (l, t), (r, b), color, 2)
            cv2.putText(frame, label, (l, t - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        if id_var:
            id_var.set(f"Suspicious IDs: {len(suspicious_ids)}")

        cv2.imshow("Detection", frame)
        if cv2.waitKey(1) == 27:
            break

    cap.release()
    cv2.destroyAllWindows()
    gui_running = False

# Start everything
threading.Thread(target=run_detection).start()
run_gui()
