import dlib, cv2, os, numpy as np, utils, random, time, pickle
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from skimage import data
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from descriptors.local_binary_pattern import LocalBinaryPatterns
from sklearn.svm import SVC
from descriptors.LocalDescriptors import WeberPattern

samples = 745
path = 'data/modalities/'

size = (256, 256)

left_eyes = [os.path.join(path, x) for x in os.listdir(path) if 'left eye' in x]
right_eyes = [os.path.join(path, x) for x in os.listdir(path) if 'right eye' in x]
mouths = [os.path.join(path, x) for x in os.listdir(path) if 'mouth' in x]
noses = [os.path.join(path, x) for x in os.listdir(path) if 'nose' in x]

train_labels = []
train_data = []

lbp = LocalBinaryPatterns(24, 8)
weber = WeberPattern()

for i in tqdm(range(samples)):
    left_eye = cv2.resize(cv2.imread(left_eyes[i], cv2.IMREAD_GRAYSCALE), size)
    right_eye = cv2.resize(cv2.imread(right_eyes[i], cv2.IMREAD_GRAYSCALE), size)
    mouth = cv2.resize(cv2.imread(mouths[i], cv2.IMREAD_GRAYSCALE), size)
    nose = cv2.resize(cv2.imread(noses[i], cv2.IMREAD_GRAYSCALE), size)

    fd_1 = weber.describe(left_eye)
    fd_2 = weber.describe(right_eye)
    fd_3 = weber.describe(mouth)
    fd_4 = weber.describe(nose)
    
    train_labels.append(1)
    train_labels.append(2)
    train_labels.append(3)
    train_labels.append(4)
    
    train_data.append(fd_1)
    train_data.append(fd_2)
    train_data.append(fd_3)
    train_data.append(fd_4)

X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size = 0.25, train_size = 0.75)

mlp = MLPClassifier((196, 256, 128))
mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_test)
y_true = y_test

print(classification_report(y_true, y_pred, target_names = ['right_eye', 'left_eye', 'mouth', 'nose']))






