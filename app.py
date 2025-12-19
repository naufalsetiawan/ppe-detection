from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, WebRtcMode
import cv2

class DummyProcessor(VideoProcessorBase):
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        cv2.putText(img, "Webcam OK", (50,50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 2)
        return frame.from_ndarray(img, format="bgr24")

webrtc_streamer(
    key="dummy",
    mode=WebRtcMode.SENDRECV,
    video_processor_factory=DummyProcessor,
    media_stream_constraints={"video": True, "audio": False}
)
