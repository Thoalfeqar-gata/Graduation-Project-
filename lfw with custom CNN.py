import cv2, numpy as np, os, sys, math
from keras.applications.resnet import ResNet50, ResNet152
from keras.applications.mobilenet_v2 import MobileNetV2
from keras.layers import Dense, Flatten
from keras.models import Sequential, Model
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.preprocessing import label_binarize
from sklearn.metrics import f1_score, precision_score, recall_score
from fusion import Fusion
import tensorflow as tf

path = 'data/lfw/lfw augmented'
checkpoint_filepath = 'data/models/checkpoints/lfw custom CNN/'
output_model_path = 'data/models/lfw custom CNN'
network_name = 'lfw custom CNN'
title = 'lfw custom CNN'
target_names = os.listdir(path)
subjects_num = len(os.listdir(path))
size = 100
images_per_subject = 45
batch_training_size = 25000
epochs = 150
patience = 30
seed = 200


model = ResNet50(include_top = False, weights = 'imagenet', input_shape = (size, size, 3))
x = Flatten()(model.layers[-1].output)
x = Dense(3000, 'relu')(x)
x = Dense(4000, 'relu')(x)
x = Dense(3000, 'relu')(x)
x = Dense(subjects_num, 'softmax')(x)
model = Model(inputs = model.input, outputs = [x])
model.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
model.summary()
try:
    model.load_weights(checkpoint_filepath)
    print('Weights loaded')
except:
    pass
callback1 = EarlyStopping(patience = patience, verbose = 1, restore_best_weights = True, monitor = 'val_accuracy')
callback2 = ModelCheckpoint(checkpoint_filepath, monitor = 'val_accuracy', verbose = 1, save_weights_only = True, save_best_only = True)

image_paths = []
training_labels = []
for i, dir in enumerate(os.listdir(path)):
    filenames = os.listdir(os.path.join(path, dir))[:images_per_subject]
    filenames = [os.path.join(path, dir, filename) for filename in filenames]
    image_paths.extend(filenames)
    training_labels.extend([i] * len(filenames))

np.random.seed(seed)
np.random.shuffle(image_paths)
np.random.seed(seed)
np.random.shuffle(training_labels)

X_testing = []
y_testing = []

batches = math.ceil(len(image_paths) / batch_training_size)
index = 0
for b in range(batches):
    print(f'Training for batch {b + 1} of {batches}...')
    with open('data/models/current batch/batch.txt', 'w') as file:
        file.write(f'Reached batch: {b}')
        
    images = []
    labels = []
    for _ in range(batch_training_size):
        if index >= len(image_paths):
            break
        
        images.append(cv2.resize(cv2.imread(image_paths[index]), (size, size)))
        labels.append(training_labels[index])
        index += 1
    images = np.array(images)
    labels = np.array(labels)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size = 0.20, train_size = 0.80, random_state = seed, shuffle = True)
    X_testing.extend(X_test)
    y_testing.extend(y_test)
    model.fit(X_train, y_train, batch_size = 32, epochs = epochs, validation_split = 0.1, callbacks = [callback1, callback2], shuffle = False, use_multiprocessing = True)
    model.save('data/models/lfw custom CNN')

prediction_probability = model.predict(np.array(X_testing))

prediction = np.argmax(prediction_probability, -1)
print(classification_report(y_testing, prediction, target_names = target_names))
fusion_obj = Fusion([], target_names)
y_bin = label_binarize(y_testing, classes = list(range(len(np.unique(training_labels)))))

fusion_obj.ROC_curve(y_bin, prediction_probability, separate_subjects = False, roc_title = f'ROC curve for {title}')
# fusion_obj.confusion_matrix(prediction, y_test, matrix_title = f'Confusion matrix for {title}')
f1 = f1_score(y_testing, prediction, average = 'weighted')
precision = precision_score(y_testing, prediction, average = 'weighted')
recall = recall_score(y_testing, prediction, average = 'weighted')
with open('data/results/results.txt', 'a') as results:
    results.write(f'Results for {title}' + f' f1 : {round(f1, 7)}, precision : {round(precision, 7)}, recall : {round(recall, 7)}\n')
