from tensorflow.keras.models import load_model
import cv2, numpy as np

model = load_model("saved_models/emotion_model.h5")
emotion_labels = ["Angry","Disgust","Fear","Happy","Sad","Surprise","Neutral"]

cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

while True:
    ret, frame = cap.read()
    if not ret: break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    for (x,y,w,h) in faces:
        face = frame[y:y+h, x:x+w]   # crop màu
        face = cv2.resize(face, (96,96))
        face = face.astype("float32")/255.0
        face = np.expand_dims(face, axis=0)

        pred = model.predict(face, verbose=0)
        label = emotion_labels[np.argmax(pred)]

        cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0), 2)
        cv2.putText(frame, label, (x,y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0,255,0), 2)

    cv2.imshow("Emotion Detection (MobileNetV2)", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
