import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import logging
import time
import sys
import os

# Global variable declarations
model = None

# Configure page
st.set_page_config(
    page_title="Emotion Classification",
    page_icon="🎭",
    layout="wide"
)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# WebRTC Configuration
RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [
        {"urls": ["stun:stun.l.google.com:19302", "stun:stun2.l.google.com:19302"]},
    ]}
)

# Define emotion classes
EMOTION_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Load the trained model
@st.cache_resource
def load_model():
    try:
        model_path = 'emotion_model.keras'
        if not os.path.exists(model_path):
            st.error(f"Model file not found at {model_path}")
            logger.error(f"Model file not found at {model_path}")
            return None
            
        model_size = os.path.getsize(model_path) / (1024 * 1024)
        logger.info(f"Model file size: {model_size:.2f} MB")
        
        model = tf.keras.models.load_model(model_path)
        logger.info("Model loaded successfully")
        return model
    except Exception as e:
        logger.error(f"Error loading model: {str(e)}")
        st.error(f"Error loading model: {str(e)}")
        return None

# Function to preprocess the image
def preprocess_image(img):
    try:
        img = cv2.resize(img, (256, 256))
        img = img / 255.0
        return img
    except Exception as e:
        logger.error(f"Error in preprocess_image: {str(e)}")
        return None

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        super().__init__()
        self.last_capture_time = time.time()
        self.connection_status = "Initializing..."
        self.model = model  # Store model reference in the instance
        try:
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            if self.face_cascade.empty():
                logger.error("Error loading face cascade classifier")
                raise ValueError("Failed to load face cascade classifier")
        except Exception as e:
            logger.error(f"Error initializing EmotionProcessor: {str(e)}")
            raise

    def on_started(self):
        self.connection_status = "Connected"
        logger.info("WebRTC connection started")

    def on_ended(self):
        self.connection_status = "Disconnected"
        logger.info("WebRTC connection ended")
        super().on_ended()

    def recv(self, frame):
        if frame is None:
            logger.warning("Received empty frame")
            return None

        if self.model is None:
            logger.error("Model not loaded")
            return frame

        current_time = time.time()
        if current_time - self.last_capture_time < 0.5:
            return frame

        self.last_capture_time = current_time

        try:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            small_gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
            faces = self.face_cascade.detectMultiScale(small_gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                x, y, w, h = x * 4, y * 4, w * 4, h * 4
                
                x = max(0, x)
                y = max(0, y)
                w = min(w, img.shape[1] - x)
                h = min(h, img.shape[0] - y)
                
                roi = img[y:y+h, x:x+w]
                if roi.size == 0:
                    continue
                    
                processed_img = preprocess_image(roi)
                if processed_img is None:
                    continue
                    
                prediction = self.model.predict(np.expand_dims(processed_img, axis=0))
                predicted_class = EMOTION_CLASSES[np.argmax(prediction)]
                confidence = float(np.max(prediction))
                
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                label = f"{predicted_class} ({confidence:.2f})"
                cv2.putText(img, label, (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
            
        except Exception as e:
            logger.error(f"Error in frame processing: {str(e)}")
            return frame

    def __del__(self):
        if hasattr(self, 'face_cascade'):
            del self.face_cascade

def main():
    st.title('Real-time Emotion Classification')
    
    # Debug information in sidebar
    if st.sidebar.checkbox("Show Debug Info"):
        st.sidebar.write("Debug Information:")
        st.sidebar.write(f"Python version: {sys.version}")
        st.sidebar.write(f"TensorFlow version: {tf.__version__}")
        st.sidebar.write(f"OpenCV version: {cv2.__version__}")
        st.sidebar.write(f"Streamlit version: {st.__version__}")
        
        if model is not None:
            st.sidebar.success("Model is loaded")
        else:
            st.sidebar.error("Model is not loaded")

    # Load model
    global model
    model = load_model()
    if model is None:
        st.error("Failed to load the emotion classification model. Please check the logs for details.")
        return

    # Display system info
    st.sidebar.subheader("System Info")
    st.sidebar.text(f"TensorFlow version: {tf.__version__}")
    st.sidebar.text(f"OpenCV version: {cv2.__version__}")

    # WebRTC streamer
    try:
        webrtc_ctx = webrtc_streamer(
            key="emotion-detection",
            video_processor_factory=EmotionProcessor,
            rtc_configuration=RTC_CONFIGURATION,
            media_stream_constraints={"video": True, "audio": False},
            async_processing=True,
        )
        
        if webrtc_ctx.state.playing:
            st.success("WebRTC connection established successfully")
        else:
            st.info("Waiting for WebRTC connection...")
            
    except Exception as e:
        st.error(f"Error initializing video stream: {str(e)}")
        logger.error(f"WebRTC initialization error: {str(e)}")
        return

    # Instructions
    st.write("1. Click 'START' to begin the webcam stream.")
    st.write("2. Allow browser access to your webcam.")
    st.write("3. The app will detect faces and classify emotions in real-time.")
    st.write("4. Click 'STOP' to end the stream.")

    # Error placeholder
    error_placeholder = st.empty()

if __name__ == "__main__":
    main()
