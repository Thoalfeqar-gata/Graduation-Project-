import cv2, joblib, numpy as np, time, face_recognition, os
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam

VideoCapture = cv2.VideoCapture(0)

dnn = VGG16(include_top = False, weights = 'imagenet', input_shape = (224, 224, 3))
for layer in dnn.layers:
    layer.trainable = False
    
predictor = Sequential([
    Dense(192, 'relu'),
    Dense(256, 'relu'),
    Dense(128, 'relu'),
    Dense(6, 'softmax')
])

predictor.compile(Adam(), 'sparse_categorical_crossentropy', ['accuracy'])
predictor.load_weights('data/models/keras model 1/')

path = os.path.join('data', 'database collage', 'detections', 'all faces with augmentation', '37')
i = 0
while True:
    ret, frame = VideoCapture.read()
    
    boxes = face_recognition.face_locations(frame, 1, 'cnn')
    
    for box in boxes:
        try:
            face = frame[box[0] - 20 : box[2] + 20, box[3] - 20 : box[1] + 20]
            face = np.array([cv2.resize(face, (224, 224))])
            
            features = np.array(dnn.predict(face, verbose = 0)).reshape(1, -1)    
            predictions = predictor.predict(features, verbose = 0)
            prediction = np.argmax(predictions, axis = -1)
            
            cv2.rectangle(frame, (box[3], box[0]), (box[1], box[2]), (0, 0, 255), 1)
            if prediction == 5:
                cv2.putText(frame, 'Thoalfeqar', (box[3], box[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
            else:
                cv2.putText(frame, 'Unknown', (box[3], box[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        except:
            continue 
    
    cv2.imshow('frames', frame)
    key =  cv2.waitKey(1)
    if key == ord('q'):
        break
    elif key == ord('a'):
        cv2.imwrite(f'Screenshot {i}.jpg', frame)
        i += 1
    
    
cv2.destroyAllWindows()
VideoCapture.release()
    