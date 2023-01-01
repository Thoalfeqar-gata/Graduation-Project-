import os, utils, pickle, face_recognition, cv2, numpy as np
from keras_vggface import VGGFace
from tqdm import tqdm
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Flatten
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping

size = 100
path = 'data/database collage/detections/DB unified/all faces with augmentation'
subjects_num = len(os.listdir(path))
training_data = []
training_labels = []
target_names = []
i = 0
for dirname, dirnames, filenames in os.walk(path):
    if len(filenames) <= 0:
        continue
    print(i)
    target_names.append(f'S{i}')
    for filename in filenames:
        image = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(dirname, filename)), (size, size)), cv2.COLOR_BGR2RGB)
        training_data.append(image)
        training_labels.append(i)
    i += 1
    
training_data = np.array(training_data)
training_labels = np.array(training_labels)
X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size = 0.25, train_size = 0.75, random_state = 200)

model = MobileNetV2(include_top = False, input_shape = (size, size, 3), weights = 'imagenet')
x = Flatten()(model.layers[-1].output)
x = Dense(384, 'relu')(x)
x = Dense(384, 'relu')(x)
x = Dense(384, 'relu')(x)
x = Dense(subjects_num, 'softmax')(x)
model = Model(inputs = model.input, outputs = [x])
model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
callback = EarlyStopping(patience = 50, verbose = 1, restore_best_weights = True, monitor = 'val_accuracy')

model.fit(X_train, y_train, 32, 250, validation_split = 0.1, shuffle = False, callbacks = [callback])
prediction_probability = model.predict(X_test)
prediction = np.argmax(prediction_probability, -1)
print(classification_report(y_test, prediction, target_names = target_names))
