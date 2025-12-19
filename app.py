import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2

class DummyProcessor(VideoProcessorBase):
    def recv(self, frame):
        # Ambil frame
        img = frame.to_ndarray(format="bgr24")
        
        # Tambah teks (BGR format)
        cv2.putText(img, "SERVER CONNECTED", (20, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        
        # WAJIB: Balikkan ke RGB agar browser tidak "stuck" loading
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        return frame.from_ndarray(img_rgb, format="rgb24")

st.title("WebRTC Debugger")

webrtc_streamer(
    key="debug",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DummyProcessor,
    # Tambahkan STUN server (Wajib untuk koneksi publik/Cloud)
    rtc_configuration={
        "iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]
    },
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    async_processing=True
)
