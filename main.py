import cv2, os, numpy as np, random, time, pickle, joblib, random, face_recognition
import utils, tensorflow as tf, keras, fusion
from descriptors.BOWDescriptors import SIFTBOWFeatures, SURFBOWFeatures
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from tqdm import tqdm
from keras_vggface import VGGFace
from skimage.feature import hog
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from descriptors.LocalDescriptors import WeberPattern, LocalBinaryPattern
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.utils import to_categorical
from descriptors.GIST import GIST
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from scipy.spatial.distance import hamming, euclidean, cosine, cityblock, minkowski, canberra, correlation, mahalanobis, dice 
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay



number_of_subjects = 10
authorised_subjects = [i for i in range(1, number_of_subjects + 1)]
lbp = LocalBinaryPattern(32, 8, (5, 5))
weber = WeberPattern((5, 5))

path = os.path.join('data', 'database collage', 'detections', 'all faces with augmentation')
subjects = [str(i) for i in range(1, number_of_subjects + 1)]

print(subjects)
size = 128

dnn = VGGFace(include_top = False, input_shape = (size, size, 3))
for layer in dnn.layers:
    layer.trainable = False

training_data = []
training_labels = []

def hog_features(images):
    features = []
    for i in tqdm(range(len(images))):
        f = hog(cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY))
        features.append(f)
    
    return np.array(features)

def face_embeddings(images):
    features = []
    for i in tqdm(range(len(images))):

        embedding = np.array(face_recognition.face_encodings(images[i], model = 'large', num_jitters = 10))
        
        if embedding.shape == (0,):
            embedding = np.zeros((128,))
        else:
            embedding = embedding[0]
        
        features.append(embedding)
    return features

samples_count = 350
for i, subject in enumerate(subjects):
    images_folder = os.path.join(path, subject)
    image_paths = [os.path.join(images_folder, image) for image in os.listdir(images_folder)][:samples_count]
    
    images = []
    for image_path in image_paths:
        image = cv2.resize(cv2.imread(image_path), (size, size))
        images.append(image)
    
    features = hog_features(images)
    training_data.extend(features)
    training_labels.extend([i] * len(features))

training_data = np.array(training_data)
training_labels = np.array(training_labels)
X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size = 0.25, train_size = 0.75, shuffle = True, random_state = 250)

model = MLPClassifier((192, 256, 128), max_iter = 10000)
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test) 
print(y_test)
target_names = [f'Subject: {subject}' for subject in authorised_subjects]
print(classification_report(y_test, np.argmax(y_pred_prob, -1), target_names = target_names))
matrix = confusion_matrix(y_test, np.argmax(y_pred_prob, -1))
display = ConfusionMatrixDisplay(matrix, display_labels = target_names)

plt.figure('Confusion matrix')
display.plot()

genuine_attempts = []
imposter_attempts = []
skip_imposter = False
for index in tqdm(range(len(authorised_subjects))):
    
    subject_data = X_test[y_test == index]
    
    condition = (y_test != index)
    for i in range(0, index): 
        condition = np.bitwise_and(condition, y_test != i)
    imposter_data = X_test[condition]
    
    genuine_proba = model.predict_proba(subject_data)
    try:
        imposter_proba = model.predict_proba(imposter_data)[:, index]
    except:
        skip_imposter = True
        
    prediction = np.argmax(genuine_proba, -1)
    actual = y_test[y_test == index]
    genuine_proba = genuine_proba[prediction == actual, index]
    
    for i in range(len(genuine_proba) - 1):
        for j in range(i, len(genuine_proba)):
            distance = 1 - np.abs(genuine_proba[i] - genuine_proba[j])
            genuine_attempts.append(distance)
    
    if not skip_imposter:
        for i in range(len(genuine_proba)):
            for j in range(len(imposter_proba)):
                distance = 1 - np.abs(genuine_proba[i] - imposter_proba[j])
                imposter_attempts.append(distance)
            
    # for i in range(len(subject_data) - 1):
    #     for j in range(i, len(subject_data)):
    #         distance = canberra(subject_data[i], subject_data[j])
    #         genuine_attempts.append(distance)
            
    # for i in range(len(subject_data)):
    #     for j in range(len(imposter_data)):
    #         distance = canberra(subject_data[i], imposter_data[j])
    #         imposter_attempts.append(distance)

genuine_attempts = np.array(genuine_attempts) * 100
imposter_attempts = np.array(imposter_attempts) * 100
bins = np.array(list(range(0, 100, 1)))

genuine_hist, _ = np.histogram(genuine_attempts, bins)
imposter_hist, _ = np.histogram(imposter_attempts, bins)

genuine_hist =  np.int64((genuine_hist / np.max(genuine_hist)) * 100)
imposter_hist = np.int64((imposter_hist / np.max(imposter_hist)) * 100)

plt.figure()
plt.plot(genuine_hist, color = 'green', label = 'Genuine score distribution')
plt.plot(imposter_hist, color = 'red', label = 'Imposter score distribution')
plt.ylabel('Distribution')
plt.xlabel('Scores')
plt.legend(loc = 'best')
plt.show()