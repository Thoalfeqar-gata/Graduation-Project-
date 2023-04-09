from deepface import DeepFace
import cv2, numpy as np, pickle, face_recognition, dlib
model = pickle.load(open('data/models/svm using dlib face embeddings/model.pickle', 'rb'))
feature_extractor = DeepFace.build_model('Facenet')
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('data/dlib data/shape_predictor_5_face_landmarks.dat')

people = {
    0 : 'Ali alezerjawy',
    1 : 'Ali hayder',
    2 : 'Humam Ali',
    3 : 'Osama Raid',
    4 : 'Zayn Alabedeen',
    5 : 'Ahmed Ali',
    6 : 'Thoalfeqar Kata',
    7 : 'Mahdy',
    8 : 'Sajad hashim',
    9 : 'Sajad',
    50 : 'Unknown'
}

cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    dets = detector(frame, 1)
    
    if len(dets) > 0:
        faces = dlib.full_object_detections()
        for det in dets:
            faces.append(shape_predictor(frame, det))
        faces = dlib.get_face_chips(frame, faces, size = 160)
        for face, det in zip(faces, dets):
            face = face[:, :, ::-1]
            face = face / 255
            face = np.expand_dims(face, axis = 0)
            feature = feature_extractor.predict(face, verbose = '0')
            prediction = model.predict(feature)[0]
            cv2.rectangle(frame, (det.left(), det.top()), (det.right(), det.bottom()), (0, 0, 255), 1)
            cv2.putText(frame, f'{people[prediction]}', (det.left(), det.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

    cv2.imshow('hi', frame)
    cv2.waitKey(1)
