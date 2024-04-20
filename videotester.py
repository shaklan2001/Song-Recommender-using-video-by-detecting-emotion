import cv2
from keras.preprocessing import image
import warnings
warnings.filterwarnings("ignore")
from keras.models import  load_model
import numpy as np
import time
from youtubesearchpython import VideosSearch
import random

blank_image = np.zeros([100, 1024, 3], dtype=np.uint8)

# load model
model = load_model("best_model.h5")

def giveSong(p):
    global blank_image
    blank_image = np.zeros([100, 1024, 3], dtype=np.uint8)
    genre = ''
    if p == 'disgusted' or p == 'angry' or p == 'afraid':
        genre = 'metal'
    elif p == 'neutral':
        genre = 'EDM'
    elif p == 'surprised':
        genre = 'indie'
    else:
        genre = p
    songType = genre + ' ' + 'songs'
    video = VideosSearch(songType, limit=10)
    videoInfo = video.result()
    rInt = random.randint(0, 9)
    recommendation = videoInfo["result"][rInt]["title"]
    cv2.putText(blank_image, recommendation, (0, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow('Recommended song', blank_image)

face_haar_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

i = 0

while True:
    ret, test_img = cap.read()  # captures frame and returns boolean value and captured image
    if not ret:
        continue
    gray_img = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)

    faces_detected = face_haar_cascade.detectMultiScale(gray_img, 1.32, 5)

    for (x, y, w, h) in faces_detected:
        cv2.rectangle(test_img, (x, y), (x + w, y + h), (255, 0, 0), thickness=7)
        roi_gray = gray_img[y:y + w, x:x + h]  # cropping region of interest i.e. face area from  image
        roi_gray = cv2.resize(roi_gray, (224, 224))
        img_pixels = image.img_to_array(roi_gray)
        img_pixels = np.expand_dims(img_pixels, axis=0)
        img_pixels /= 255

        predictions = model.predict(img_pixels)

        # find max indexed array
        max_index = np.argmax(predictions[0])

        emotions = ('angry', 'disgusted', 'afraid', 'happy', 'sad', 'surprised', 'neutral')
        predicted_emotion = emotions[max_index]

        cv2.putText(test_img, predicted_emotion, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if i % 50 == 0:
            if i != 0:
                cv2.destroyWindow('Recommended song')
            giveSong(predicted_emotion)

    resized_img = cv2.resize(test_img, (1000, 700))
    cv2.imshow('Facial emotion analysis ', resized_img)
    cv2.imshow('Recommended song', blank_image)

    if cv2.waitKey(10) == ord('q'):  # wait until 'q' key is pressed
        break
    time.sleep(0.01)
    i+=1

cap.release()
cv2.destroyAllWindows