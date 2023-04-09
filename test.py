import cv2, face_recognition, numpy as np, tensorflow as tf

def to_opencv_bounding_box(location):
    top, right, bottom, left = location
    return (left, top, right - left, bottom - top)
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
# people = {
#     0 : 'Dr. Saadoon',
#     1 : 'Dr. Muayad',
#     2 : 'Muhammed (student)',
#     3 : 'blank for now',
#     4 : 'Miss Yasameen',
#     5 : 'blank for now',
#     6 : 'clean worker (woman)',
#     7 : 'Dr. Ahmad Saeed',
#     8 : 'blank for now',
#     9 : 'blank for now',
#     10 : 'Dr. Walaa',
#     11 : 'blank for now',
#     12 : 'Dr. Muhammed Ali',
#     13 : 'Salman',
#     14 : 'Dr. Raoof',
#     15 : 'blank for now',
#     16 : 'blank for now',
#     17 : 'blank for now', 
#     18 : 'Dr. Nidhal',
#     19 : 'blank for now',
#     20 : 'Dr. Dhafer',
#     21 : 'Dr. Sarmad',
#     22 : 'blank for now',
#     23 : 'blank for now',
#     24 : 'blank for now',
#     25 : 'blank for now',
#     26 : 'Dr. Sawsan',
#     27 : 'Dr. Mahmood Zaky',
#     28 : 'Engineer',
#     29 : 'Dr. Yasameen',
#     30 : 'Miss Zina',
#     31 : 'Miss Hadeel',
#     32 : 'Miss Rana', 
#     33 : 'Dr. Basma',
#     34 : 'blank for now',
#     35 : 'Dr. Fatima',
#     36 : 'blank for now',
#     37 : 'blank for now',
#     38 : 'blank for now',
#     39 : 'Dr. Yaarob', 
#     40 : 'Dr. Ez',
#     41 : 'blank for now',
#     42 : 'Thoalfeqar',
#     43 : 'Ahmad Ali',
#     44 : 'Ali Hayder',
#     45 : 'Humam Ali',
#     46 : 'Mahdy', 
#     47 : 'Osama Raaid',
#     48 : 'Sajad 1',
#     49 : 'Sajad 2',
#     50 : 'Unknown'
#     }

'''
model_type can be either ['tflite', 'tf', 'svm']
face_detector can be eitehr ['hog', 'opencv']
'''
def run_face_recognition(face_detector = 'hog', frame_size = (640, 480), apply_clahe = True, model_path = 'data/models/MobileNetV2 128 optimized/model.tflite', model_type = 'tflite', face_size = (180, 180), threshold = 0.9999): 
    video_capture = cv2.VideoCapture(0)
    video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, frame_size[0])
    video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_size[1])
    clahe = cv2.createCLAHE(clipLimit = 6, tileGridSize = (6, 6))
    opencv_face_detector = cv2.CascadeClassifier('data/opencv data/cascades/haarcascade_frontalface_default.xml')
    
    if model_type == 'tflite':
        model = tf.lite.Interpreter(model_path = model_path)
        model.allocate_tensors()
        input_details = model.get_input_details()
        output_details = model.get_output_details()
    elif model_type == 'tf':
        model = tf.keras.models.load_model(filepath = model_path)
        for layer in model.layers:
            layer.trainable = False
            
    while True:
        ret, frame = video_capture.read()
        frame = cv2.resize(frame, frame_size)
        
        if apply_clahe:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
            frame[:, :, 0] = clahe.apply(frame[:, :, 0])
            frame = cv2.cvtColor(frame, cv2.COLOR_LAB2BGR)
        
        if face_detector == 'hog':
            locations = face_recognition.face_locations(frame, 1, 'hog')
            for i in range(len(locations)):
                locations[i] = to_opencv_bounding_box(locations[i])
        elif face_detector == 'opencv':
            locations = opencv_face_detector.detectMultiScale(frame, 1.1, 15)
        elif face_detector == 'cnn':
            locations = face_recognition.face_locations(frame, 1, 'cnn')
            for i in range(len(locations)):
                locations[i] = to_opencv_bounding_box(locations[i])

        for x, y, w, h in locations:
            face = cv2.resize(frame[y : y + h, x : x + w], face_size)
            input_image = np.expand_dims(face, axis = 0).astype(np.float32)
            
            if model_type == 'tflite':
                model.set_tensor(input_details[0]['index'], input_image)
                model.invoke()
                prediction = model.get_tensor(output_details[0]['index'])[0]
                p = int(np.argmax(prediction, -1))
                if prediction[p] < threshold:
                    prediction = 50
                else:
                    prediction = p
            elif model_type == 'tf':
                prediction = model.predict(input_image, verbose = '0')[0]
                p = int(np.argmax(prediction, -1))
                if prediction[p] < threshold:
                    prediction = 50
                else:
                    prediction = p

            cv2.putText(frame, people[prediction], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 1)
        cv2.imshow('hi', frame)
        
        if cv2.waitKey(1) == ord('q'):
            break
    
model_path = 'data/models/NASNet (128-128-128) optimized/model.tflite'
run_face_recognition(apply_clahe = True, face_detector = 'cnn', model_path = model_path, model_type = 'tflite', frame_size = (640, 480))
            
        
        
            
        
            