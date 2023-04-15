from deepface import DeepFace
import cv2, numpy as np, pickle, face_recognition, dlib, tensorflow as tf, utils, time
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
    -1 : 'Unknown'
}

'''
classifier_types = [tf, svm, tflite]
feature_extractor_type = [tf, tflite]
face_detector = [opencv, hog, cnn]
'''
def main(classifier_path, feature_extractor_path, face_detector = 'hog', classifier_type = 'tf', feature_extractor_type = 'tf', threshold = 0.98, show_fps = True):
    if classifier_type == 'tf':
        classifier = tf.keras.models.load_model(filepath = classifier_path)
    elif classifier_type == 'tflite':
        classifier = tf.lite.Interpreter(model_path = classifier_path)
        classifier.allocate_tensors()
        classifier_input_details = classifier.get_input_details()
        classifier_output_details = classifier.get_output_details()
    elif classifier_type == 'svm':
        classifier = pickle.load(open(classifier_path, 'rb'))
        
    if feature_extractor_type == 'tf':
        feature_extractor = DeepFace.build_model('Facenet')
    elif feature_extractor_type == 'tflite':
        feature_extractor = tf.lite.Interpreter(model_path = feature_extractor_path)
        feature_extractor.allocate_tensors()
        feature_extractor_input_details = feature_extractor.get_input_details()
        feature_extractor_output_details = feature_extractor.get_output_details()
    
    if face_detector == 'opencv':
        detector = cv2.CascadeClassifier('data/opencv data/cascades/haarcascade_frontalface_default.xml')

    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        if show_fps:
            t1 = time.perf_counter()
            
        ret, frame = cap.read()
        if face_detector == 'opencv':
            dets = detector.detectMultiScale(frame, 1.1, 20)
        elif face_detector == 'hog':
            dets = [utils.to_opencv_bounding_box(det) for det in  face_recognition.face_locations(frame, 1, 'hog')]
        elif face_detector == 'cnn':
            dets = [utils.to_opencv_bounding_box(det) for det in  face_recognition.face_locations(frame, 1, 'cnn')]
            

        if len(dets) > 0:
            faces = utils.preprocess_image(frame, dets)
            
            for face, det in zip(faces, dets):
                face = np.expand_dims(face, axis = 0)
                if feature_extractor_type == 'tf':
                    feature = feature_extractor.predict(face, verbose = '0')
                elif feature_extractor_type == 'tflite':      
                    feature_extractor.set_tensor(feature_extractor_input_details[0]['index'], face.astype(np.float32))
                    feature_extractor.invoke()
                    feature = feature_extractor.get_tensor(feature_extractor_output_details[0]['index'])
                
                if classifier_type == 'tf':
                    prediction = classifier.predict(feature, verbose = '0')[0]
                    p = np.argmax(prediction, -1)
                    if prediction[p] < threshold:
                        p = -1
                elif classifier_type == 'tflite':
                    classifier.set_tensor(classifier_input_details[0]['index'], feature)
                    classifier.invoke()
                    prediction = classifier.get_tensor(classifier_output_details[0]['index'])[0]
                    p = np.argmax(prediction, -1)
                    if prediction[p] < threshold:
                        p = -1
                elif classifier_type == 'svm':
                    prediction = classifier.predict_proba(feature)[0]
                    p = np.argmax(prediction, -1)
                    if prediction[p] < threshold:
                        p = -1
                        
                x, y, w, h = det
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
                cv2.putText(frame, f'{people[p]}', (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 255), 2)


        if show_fps:
            t2 = time.perf_counter()
            fps = np.round(1/(t2 - t1), 3)
            cv2.putText(frame, f'{fps} fps', (0, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
            cv2.putText(frame, f'{fps} fps', (0, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            
        cv2.imshow('hi', frame)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

classifier_path = 'data/models/svm using dlib face embeddings/model.pickle'
feature_extractor_path = 'data/models/Neural network using facenet embeddings optimized/feature extractor/model.tflite'
main(classifier_path, feature_extractor_path, 'cnn', classifier_type = 'svm', feature_extractor_type = 'tflite')
    
    