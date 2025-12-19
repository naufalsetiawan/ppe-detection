import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from streamlit_webrtc import webrtc_streamer, WebRtcMode, VideoProcessorBase

# --- LOGIKA GEOMETRI ---
def is_inside(box_small, box_big):
    xA = max(box_small[0], box_big[0])
    yA = max(box_small[1], box_big[1])
    xB = min(box_small[2], box_big[2])
    yB = min(box_small[3], box_big[3])
    interArea = max(0, xB - xA) * max(0, yB - yA)
    smallArea = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1])
    return (interArea / float(smallArea + 1e-6)) > 0.8

def process_frame(frame, model):
    # Optimasi: imgsz=320 membantu CPU memproses lebih cepat
    results = model(frame, conf=0.3, imgsz=320, verbose=False)[0]
    person_boxes, ppe_items = [], []

    for box in results.boxes:
        cls = int(box.cls[0])
        label = results.names[cls]
        coords = list(map(int, box.xyxy[0]))
        if label == "person":
            person_boxes.append(coords)
        else:
            ppe_items.append({"label": label, "box": coords})
            color_item = (0, 255, 255) if label == "helmet" else (255, 255, 0)
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color_item, 1)

    for p_box in person_boxes:
        has_helmet = any(is_inside(item["box"], p_box) for item in ppe_items if item["label"] == "helmet")
        has_vest = any(is_inside(item["box"], p_box) for item in ppe_items if item["label"] == "vest")
        color, status = ((0, 255, 0), "SAFE") if (has_helmet and has_vest) else ((0, 0, 255), "DANGER")
        cv2.rectangle(frame, (p_box[0], p_box[1]), (p_box[2], p_box[3]), color, 2)
        cv2.putText(frame, status, (p_box[0], p_box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
    return frame

# --- CLASS PROCESSOR (Pastikan nama ini SAMA dengan yang dipanggil di bawah) ---
class PPEVideoProcessor(VideoProcessorBase):
    def __init__(self):
        # Load model di dalam init agar hanya jalan sekali
        self.model = YOLO("best.pt")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        processed = process_frame(img, self.model)
        # WebRTC butuh format RGB untuk ditampilkan ke browser
        return frame.from_ndarray(cv2.cvtColor(processed, cv2.COLOR_BGR2RGB), format="rgb24")

# --- UI ---
st.title("🛡️ PPE Detection System (Cloud Optimized)")

webrtc_streamer(
    key="ppe-detection",
    mode=WebRtcMode.SENDRECV,
    # Panggil Class yang sudah kita buat di atas
    video_processor_factory=PPEVideoProcessor,
    async_processing=True,
    rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
    media_stream_constraints={
        "video": {
            "width": {"ideal": 320}, 
            "height": {"ideal": 240},
            "frameRate": {"ideal": 10}
        },
        "audio": False
    }
)
