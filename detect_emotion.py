# detect_emotion.py
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array

model = load_model("model/emotion_model.h5")
labels = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)

    results = []

    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        try:
            roi = cv2.resize(roi_gray, (48, 48))
        except:
            continue

        roi = roi.astype("float") / 255.0
        roi = img_to_array(roi)
        roi = np.expand_dims(roi, axis=0)

        preds = model.predict(roi, verbose=0)[0]
        label = labels[np.argmax(preds)]
        confidence = float(np.max(preds)) * 100

        results.append({
            "box": (x, y, w, h),
            "label": label,
            "confidence": confidence
        })

    return results
