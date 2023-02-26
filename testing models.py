import cv2, numpy as np, face_recognition, keras, time
from skimage.feature import hog
model = keras.models.load_model('data/models/hog with feature fusion')

model.summary()
for layer in model.layers:
    layer.trainable = False

videoCapture = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('data/opencv data/cascades/haarcascade_frontalface_default.xml')
while True:
    t1 = time.time()
    ret, frame = videoCapture.read()
    boxes = face_detector.detectMultiScale(frame, 1.1, 10)
    for x, y, w, h in boxes:
        face = cv2.cvtColor(cv2.resize(frame[y : y + h, x : x + w], (100, 100)), cv2.COLOR_BGR2GRAY)
        features = hog(face, 10, (8, 8), (2, 2))
        features = np.expand_dims(features, 0)
        prediction = model.predict(features, verbose = 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        if np.argmax(prediction, -1) == 42:
            cv2.putText(frame, 'Thoalfeqar', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(prediction[0][42])
        else:
            cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            print(np.argmax(prediction, -1))
            
        
    cv2.imshow('hi', frame)
    if cv2.waitKey(1) == ord('q'):
        break
    