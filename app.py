import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from PIL import Image
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase
import av

def is_inside(box_small, box_big):
    # IoA (Intersection over Area)

    # shape : box[min_x,min_y,max_x,max_y]
    # cek titik terdalam
    xA = max(box_small[0], box_big[0])
    yA = max(box_small[1], box_big[1])
    xB = min(box_small[2], box_big[2])
    yB = min(box_small[3], box_big[3])
    interArea = max(0, xB - xA) * max(0, yB - yA) # Luas Area Intersect

    smallArea = (box_small[2] - box_small[0]) * (box_small[3] - box_small[1]) # Luas = Lebar * Tinggi
    
    return (interArea / float(smallArea + 1e-6)) > 0.8 # hitung proporsi


def process_frame(frame, model):
    results = model(frame, conf=0.3)[0]

    # ============================================
    # results: object hasil deteksi untuk SATU frame
    # ├─ .boxes    -> daftar semua bounding box yang terdeteksi di frame
    # │   ├─ box.xyxy  -> koordinat kotak [x1, y1, x2, y2], shape (1,4)
    # │   ├─ box.cls   -> ID kelas objek, shape (1,), misal 0=person, 1=helmet
    # │   └─ box.conf  -> confidence deteksi, shape (1,), misal 0.95
    #
    # ├─ .names    -> mapping ID kelas ke label nama, misal {0:'person', 1:'helmet', 2:'vest'}
    # ============================================
    
    person_boxes = []
    ppe_items = [] 

    # klasifikasi
    for box in results.boxes: # loop bounding box
        cls = int(box.cls[0]) 
        label = results.names[cls]
        coords = list(map(int, box.xyxy[0]))
        
        if label == "person":
            person_boxes.append(coords) # person
        else:
            ppe_items.append({"label": label, "box": coords}) # PPE
            color_item = (0, 255, 255) if label == "helmet" else (255, 255, 0) # set color
            cv2.rectangle(frame, (coords[0], coords[1]), (coords[2], coords[3]), color_item, 1) # box
            cv2.putText(frame, label, (coords[0], coords[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color_item, 1) # text

    # cek PPE per orang
    for p_box in person_boxes: # loop person
        px1, py1, px2, py2 = p_box
        has_helmet = any(is_inside(item["box"], p_box) for item in ppe_items if item["label"] == "helmet")
        has_vest = any(is_inside(item["box"], p_box) for item in ppe_items if item["label"] == "vest")
        
        if has_helmet and has_vest:
            color, status = (0, 255, 0), "SAFE"
        else:
            color, status = (0, 0, 255), "DANGER: INCOMPLETE PPE"
            
        # person box
        cv2.rectangle(frame, (px1, py1), (px2, py2), color, 2)
        cv2.putText(frame, status, (px1, py1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    return frame

# ================= VIDEO PROCESSOR =================
class VideoProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = YOLO("best.pt")

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = process_frame(img, self.model)
        return av.VideoFrame.from_ndarray(img, format="bgr24")
        

# ================= UI =================
st.set_page_config(page_title="Real-Time PPE Detection")
st.title("Real-Time PPE Detection System")

webrtc_streamer(
    key="ppe",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=lambda: VideoProcessor(),
    media_stream_constraints={"video": True, "audio": False},
    client_settings=ClientSettings(
        rtc_configuration={
            "iceServers": [
                {
                    "urls": [
                        "stun:global.stun.twilio.com:3478",
                        "turn:global.turn.twilio.com:3478?transport=udp",
                        "turn:global.turn.twilio.com:3478?transport=tcp",
                    ],
                    "username": st.secrets["TWILIO_ACCOUNT_SID"],
                    "credential": st.secrets["TWILIO_AUTH_TOKEN"],
                }
            ]
        }
    ),
)










