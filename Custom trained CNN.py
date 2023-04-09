import os, utils, pickle, face_recognition, cv2, numpy as np
from keras_vggface import VGGFace
from tqdm import tqdm
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.nasnet import NASNetMobile
from keras.applications.vgg16 import VGG16
from keras.applications.resnet import ResNet50
from keras_vggface import VGGFace
from keras.layers import Dense, Flatten
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, precision_score, recall_score
from fusion import Fusion
import tensorflow as tf

size = 180
network_name = 'NASNet (128-128-128)'
path = 'data/database collage/detections/DB unified of friends/DB with augmentation'
checkpoint_filepath = f'data/models/checkpoints/{network_name}/'
subjects_num = len(os.listdir(path))
training_data = []
training_labels = []
target_names = []
limit = 1000000
with open('data/models/model input sizes.txt', 'a') as file:
    file.writelines(f'{network_name} = ({size}, {size})')
    
for dirname, dirnames, filenames in os.walk(path):
    if len(filenames) <= 0:
        continue
    i = os.path.split(dirname)[1]
    print(i)
    target_names.append(f'S{i}')
    counter = 0
    for filename in filenames:
        image = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(dirname, filename)), (size, size)), cv2.COLOR_BGR2RGB)
        training_data.append(image)
        training_labels.append(int(i))
        counter += 1
        if counter >= limit:
            break
    
training_data = np.array(training_data)
training_labels = np.array(training_labels)
X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size = 0.20, train_size = 0.80, random_state = 200)

model = NASNetMobile(include_top = False, input_shape = (size, size, 3))
x = Flatten()(model.layers[-1].output)
x = Dense(128, 'relu')(x)
x = Dense(128, 'relu')(x)
x = Dense(128, 'relu')(x)
x = Dense(subjects_num, 'softmax')(x)
model = Model(inputs = model.input, outputs = [x])
model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
model.summary()

callback1 = EarlyStopping(patience = 20, verbose = 1, restore_best_weights = True, monitor = 'val_accuracy')
callback2 = ModelCheckpoint(checkpoint_filepath, monitor = 'val_accuracy', verbose = 1, save_weights_only = True)
try:
    model.load_weights(checkpoint_filepath)
    print('Weights loaded')
except:
    pass

# model.fit(X_train, y_train, 32, 500, validation_split = 0.1, shuffle = False, callbacks = [callback1, callback2])
prediction_probability = model.predict(X_test)
model.save(f'data/models/{network_name}')
model_converter = tf.lite.TFLiteConverter.from_keras_model(model)
model_optimized = model_converter.convert()
with open(f'data/models/{network_name} optimized/model.tflite', 'wb') as f:
  f.write(model_optimized)

interpreter = tf.lite.Interpreter(model_path = f'data/models/{network_name} optimized/model.tflite')
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
X_test = np.array(X_test).astype(np.float32)
prediction_probability_optimized = []
for i in range(len(X_test)):
    interpreter.set_tensor(input_details[0]['index'], [X_test[i]])
    interpreter.invoke()
    pred = interpreter.get_tensor(output_details[0]['index'])
    prediction_probability_optimized.append(pred)
prediction_probability_optimized = np.array(prediction_probability_optimized)

titles = [f'{network_name}', f'{network_name} optimized']
for i, prediction_probabilities in enumerate([prediction_probability, prediction_probability_optimized]):
    prediction = np.argmax(prediction_probabilities, -1)
    print(classification_report(y_test, prediction, target_names = target_names))
    fusion_obj = Fusion([], [f'S{i}' for i in range(subjects_num)])
    y_bin = label_binarize(y_test, classes = list(range(len(np.unique(training_labels)))))

    fusion_obj.ROC_curve(y_bin, prediction_probabilities, separate_subjects = False, roc_title = f'ROC curve for {titles[i]}')
    fusion_obj.confusion_matrix(prediction, y_test, matrix_title = f'Confusion matrix for {titles[i]}')
    f1 = f1_score(y_test, prediction, average = 'weighted')
    precision = precision_score(y_test, prediction, average = 'weighted')
    recall = recall_score(y_test, prediction, average = 'weighted')
    with open('data/results/results.txt', 'a') as results:
        results.write(f'Results for {titles[i]}' + f' f1 : {round(f1, 5)}, precision : {round(precision, 5)}, recall : {round(recall, 5)}\n')
