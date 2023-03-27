import cv2, numpy as np, face_recognition, keras, time
model = keras.models.load_model('data/models/MobileNetV2 128')
model.summary()
for layer in model.layers:
    layer.trainable = False

videoCapture = cv2.VideoCapture(0)
face_detector = cv2.CascadeClassifier('data/opencv data/cascades/haarcascade_frontalface_default.xml')
times = []
frames_processed = 0
while True:
    t1 = time.time()
    ret, frame = videoCapture.read()
    boxes = face_detector.detectMultiScale(frame, 1.1, 10)
    for x, y, w, h in boxes:
        face = cv2.resize(frame[y : y + h, x : x + w], (100, 100))
        prediction = model.predict(np.expand_dims(face, axis = 0), verbose = 0)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        if np.argmax(prediction, -1) == 42:
            cv2.putText(frame, 'Thoalfeqar', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, 'Unknown', (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        
    t2 = time.time()
    cv2.imshow('hi', frame)
    times.append(1/(t2 - t1))
    frames_processed += 1
    
    if cv2.waitKey(1) == ord('q'):
        break
    if frames_processed >= 10000:
        break

print(np.average(times))