import os, utils, pickle, face_recognition, cv2, numpy as np
from keras_vggface import VGGFace
from tqdm import tqdm
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.applications.vgg16 import VGG16
from keras.layers import Dense, Flatten
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, precision_score, recall_score
from fusion import Fusion

size = 100
path = 'data/database collage/detections/DB unified/all faces with augmentation'
subjects_num = len(os.listdir(path))
training_data = []
training_labels = []
target_names = []
for dirname, dirnames, filenames in os.walk(path):
    if len(filenames) <= 0:
        continue
    i = os.path.split(dirname)[1]
    print(i)
    target_names.append(f'S{i}')
    for filename in filenames:
        image = cv2.cvtColor(cv2.resize(cv2.imread(os.path.join(dirname, filename)), (size, size)), cv2.COLOR_BGR2RGB)
        training_data.append(image)
        training_labels.append(int(i))
    
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
model.save('data/models/MobileNetV2 with me in it')
prediction_probability = model.predict(X_test)
prediction = np.argmax(prediction_probability, -1)
print(classification_report(y_test, prediction, target_names = target_names))
fusion_obj = Fusion([], [f'S{i}' for i in range(subjects_num)])
y_bin = label_binarize(y_test, classes = list(range(len(np.unique(training_labels)))))

fusion_obj.ROC_curve(y_bin, prediction_probability, separate_subjects = False, roc_title = 'ROC curve for MobileNetV2 (with me in it)')
fusion_obj.confusion_matrix(prediction, y_test, matrix_title = 'Confusion matrix for MobileNetV2 (with me in it)')
f1 = f1_score(y_test, prediction, average = 'weighted')
precision = precision_score(y_test, prediction, average = 'weighted')
recall = recall_score(y_test, prediction, average = 'weighted')
with open('data/results/results.txt', 'a') as results:
    results.write('Results for MobileNetV2 (with me in it)' + f' f1 : {round(f1, 5)}, precision : {round(precision, 5)}, recall : {round(recall, 5)}\n')
