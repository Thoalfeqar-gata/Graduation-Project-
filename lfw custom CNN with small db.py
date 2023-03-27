import cv2, numpy as np, os, sys, math
from keras.applications.resnet import ResNet50, ResNet152
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Flatten
from keras.models import Sequential, Model
from keras import optimizers
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, precision_score, recall_score
from fusion import Fusion
import tensorflow as tf

path = 'data/pins dataset/pins dataset augmented'
checkpoint_filepath = 'data/models/checkpoints/pins dataset/'
output_model_path = 'data/models/pins dataset custom CNN'
network_name = 'pins dataset custom CNN'
title = 'pins dataset custom CNN'
target_names = os.listdir(path)
subjects_num = len(os.listdir(path))
size = 100
images_per_subject = 1000
epochs = 1000
patience = 100
seed = 200


model = MobileNetV2(include_top = False, weights = 'imagenet', input_shape = (size, size, 3))
x = Flatten()(model.layers[-1].output)
x = Dense(768, 'relu')(x)
x = Dense(512, 'relu')(x)
x = Dense(256, 'relu')(x)
x = Dense(subjects_num, 'softmax')(x)
model = Model(inputs = model.input, outputs = [x])
model.compile(optimizers.RMSprop(momentum = 0.9), 'sparse_categorical_crossentropy', ['accuracy'])
model.summary()

callback1 = EarlyStopping(patience = patience, verbose = 1, restore_best_weights = True, monitor = 'val_accuracy')
callback2 = ModelCheckpoint(checkpoint_filepath, monitor = 'val_accuracy', verbose = 1, save_weights_only = True, save_best_only = True)

image_paths = []
training_labels = []
for i, dir in enumerate(os.listdir(path)):
    filenames = os.listdir(os.path.join(path, dir))[:images_per_subject]
    filenames = [os.path.join(path, dir, filename) for filename in filenames]
    image_paths.extend(filenames)
    training_labels.extend([i] * len(filenames))

training_data = []
for image_path in image_paths:
    image = cv2.resize(cv2.imread(image_path), (size, size))
    training_data.append(image)

training_data = np.array(training_data)
training_labels = np.array(training_labels)
X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size = 0.2, train_size = 0.8, random_state = seed, shuffle = True)

model.fit(X_train, y_train, batch_size = 32, epochs = epochs, validation_split = 0.1, callbacks = [callback1, callback2], shuffle = False, use_multiprocessing = True)
prediction_probability = model.predict(np.array(X_test))

prediction = np.argmax(prediction_probability, -1)
print(classification_report(y_test, prediction, target_names = target_names))
fusion_obj = Fusion([], target_names)
y_bin = label_binarize(y_test, classes = list(range(len(np.unique(training_labels)))))

fusion_obj.ROC_curve(y_bin, prediction_probability, separate_subjects = False, roc_title = f'ROC curve for {title}')
# fusion_obj.confusion_matrix(prediction, y_test, matrix_title = f'Confusion matrix for {title}')
f1 = f1_score(y_test, prediction, average = 'weighted')
precision = precision_score(y_test, prediction, average = 'weighted')
recall = recall_score(y_test, prediction, average = 'weighted')
with open('data/results/results.txt', 'a') as results:
    results.write(f'Results for {title}' + f' f1 : {round(f1, 7)}, precision : {round(precision, 7)}, recall : {round(recall, 7)}\n')
