import cv2, os, numpy as np, random, time, pickle, joblib, random
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



number_of_subjects = 37
authorised_subjects = [i for i in range(1, number_of_subjects + 1)]
lbp = LocalBinaryPattern(32, 8, (5, 5))

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

samples_count = 350
for i, subject in enumerate(subjects):
    images_folder = os.path.join(path, subject)
    image_paths = [os.path.join(images_folder, image) for image in os.listdir(images_folder)][:samples_count]
    
    images = []
    for image_path in image_paths:
        image = cv2.resize(cv2.imread(image_path), (size, size))
        images.append(image)
    
    features = np.array(dnn.predict(np.array(images))).reshape(len(images), -1)
    training_data.extend(features)
    training_labels.extend([i] * len(features))

training_data = np.array(training_data)
training_labels = np.array(training_labels)
X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size = 0.25, train_size = 0.75, shuffle = True, random_state = 250)

model = OneVsRestClassifier(SVC(kernel = 'linear', verbose = True, max_iter = 10000, probability = True), n_jobs = -1)
model.fit(X_train, y_train)
y_pred_prob = model.predict_proba(X_test) 


target_names = [f'Subject: {subject}' for subject in authorised_subjects]
print(classification_report(y_test, np.argmax(y_pred_prob, -1), target_names = target_names))

genuine_attempts = []
imposter_attempts = []
for subject in authorised_subjects:
    index = subject - 1
    
    subject_data = X_test[y_test == index]
    imposter_data = X_test[y_test != index]
    
    
    y_genuine_probability = model.predict_proba(subject_data)
    prediction = np.argmax(y_genuine_probability, 1)
    actual = y_test[y_test == index]
    y_genuine_probability = y_genuine_probability[prediction == actual]
    
    y_imposter_probability = model.predict_proba(imposter_data)[:len(y_genuine_probability)]
    
    genuine_score = y_genuine_probability[:, index]
    genuine_attempts.extend(genuine_score)

    imposter_score = y_imposter_probability[:, index]
    imposter_attempts.extend(imposter_score)
    
    

genuine_attempts = np.array(genuine_attempts) * 100
imposter_attempts = np.array(imposter_attempts) * 100
bins = np.array(list(range(0, 100, 1)))

plot_genuine = []
plot_imposter = []

for i in range(100):
    count_genuine = 0
    count_imposter = 0
    
    for genuine_attempt in genuine_attempts:
        if genuine_attempt <= i:
            count_genuine += 1
    
    for imposter_attempt in imposter_attempts:
        if imposter_attempt >= i:
            count_imposter += 1
    
    plot_genuine.append(count_genuine)
    plot_imposter.append(count_imposter)
    
    
plt.figure()
plt.plot(plot_imposter, color = 'red', label = 'FAR')
plt.plot(plot_genuine, color = 'green', label = 'FRR')
plt.legend(loc = 'best')
plt.xlim([0, 100])
plt.show()



