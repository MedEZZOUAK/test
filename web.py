import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import os
from pyngrok import ngrok
import av
import logging
import time

# Set up logging
logging.basicConfig(level=logging.INFO)
# Load the trained model
@st.cache_resource
def load_model():
    try:
        model = tf.keras.models.load_model('emotion_model.keras')
        logging.info("Model loaded successfully")
        return model
    except Exception as e:
        logging.error(f"Error loading model: {str(e)}")
        return None

model = load_model()
if model is None:
    st.error("Failed to load the model. Please check the model file and path.")

# Define emotion classes
emotion_classes = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']

# Function to preprocess the image
def preprocess_image(img):
    img = cv2.resize(img, (256, 256))  # Update to match model's expected input size
    img = img / 255.0  # Normalize
    return img

class EmotionProcessor(VideoProcessorBase):
    def __init__(self):
        self.last_capture_time = time.time()
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    def recv(self, frame):
        current_time = time.time()
        if current_time - self.last_capture_time < 0.1:  # Capture every 0.5 seconds
            return frame

        self.last_capture_time = current_time

        try:
            img = frame.to_ndarray(format="bgr24")
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            
            # Resize for faster face detection
            small_gray = cv2.resize(gray, (0, 0), fx=0.25, fy=0.25)
            faces = self.face_cascade.detectMultiScale(small_gray, 1.1, 4)

            for (x, y, w, h) in faces:
                # Scale coordinates back up
                x, y, w, h = x * 4, y * 4, w * 4, h * 4
                roi = img[y:y+h, x:x+w]
                processed_img = preprocess_image(roi)
                prediction = model.predict(np.expand_dims(processed_img, axis=0))
                predicted_class = emotion_classes[np.argmax(prediction)]
                logging.info(f"Predicted class: {predicted_class}")
                cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
                cv2.putText(img, predicted_class, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36, 255, 12), 2)

            return av.VideoFrame.from_ndarray(img, format="bgr24")
        except Exception as e:
            logging.error(f"Error in EmotionProcessor: {str(e)}")
            return frame

# Streamlit app
st.title('Real-time Emotion Classification')

webrtc_ctx = webrtc_streamer(
    key="emotion-detection",
    video_processor_factory=EmotionProcessor,
    rtc_configuration=RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}),
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)

st.write("1. Click 'START' to begin the webcam stream.")
st.write("2. Allow browser access to your webcam.")
st.write("3. The app will detect faces and classify emotions in real-time.")
st.write("4. Click 'STOP' to end the stream.")

# Display system info
st.sidebar.subheader("System Info")
st.sidebar.text(f"TensorFlow version: {tf.__version__}")
st.sidebar.text(f"OpenCV version: {cv2.__version__}")

# Add a placeholder for error messages
error_placeholder = st.empty()

# Function to display errors
def show_error(error_message):
    error_placeholder.error(error_message)

# Error handling for model loading
if model is None:
    show_error("Failed to load the model. Please check the model file and path.")
