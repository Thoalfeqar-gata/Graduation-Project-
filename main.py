import cv2, os, numpy as np, random, time, pickle, joblib
import utils, tensorflow as tf, keras, fusion
from descriptors.BOWDescriptors import SIFTBOWFeatures, SURFBOWFeatures
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from tqdm import tqdm
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



number_of_subjects = 15
authorised_subjects = [1, 5, 7]

path = os.path.join('data', 'database collage', 'detections', 'all faces with augmentation')
subjects = [str(i) for i in range(number_of_subjects)]

print(subjects)
size = 224

dnn = VGG16(include_top = False, weights = 'imagenet', input_shape = (size, size, 3))
for layer in dnn.layers:
    layer.trainable = False
print(dnn.summary())
input()
training_data = []
training_labels = []

def hog_features(images):
    features = []
    for i in tqdm(range(len(images))):
        f = hog(images[i])
        features.append(f)
    
    return np.array(features)

i = 1
for subject in subjects:
    images_folder = os.path.join(path, subject)
    image_paths = [os.path.join(images_folder, image) for image in os.listdir(images_folder)]
    
    images = []
    for image_path in image_paths:
        image = cv2.resize(cv2.imread(image_path), (size, size))
        images.append(image)
    
    features = np.array(dnn.predict(np.array(images))).reshape(len(images), -1)
    
    training_data.extend(features)
    
    if int(subject) in authorised_subjects:
        training_labels.extend([i] * len(features))
        i = i + 1
    else:
        training_labels.extend([0] * len(features))

training_data = np.array(training_data)
training_labels = np.array(training_labels)
X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size = 0.25, train_size = 0.75, shuffle = True, random_state = 250)

model = Sequential([
    Input(shape = (training_data.shape[1])), 
    Dense(192, 'relu'),
    Dense(256, 'relu'),
    Dense(128, 'relu'),
    Dense(len(authorised_subjects) + 1, 'softmax')
])

model.compile(Adam(), 'sparse_categorical_crossentropy', ['accuracy'])
model.fit(X_train, y_train, shuffle = False, use_multiprocessing = True, epochs = 75, verbose = 1)

y_pred_prob = model.predict(X_test) 
y_bin = label_binarize(y_test, classes = list(range(len(authorised_subjects) + 1)))

line_styles = [':', '-', '--', '-.']
target_names = ['Subject: Unknown']
target_names.extend([f'Subject: {subject}' for subject in authorised_subjects])
print(classification_report(y_test, np.argmax(y_pred_prob, -1), target_names = target_names))

for i in range(len(authorised_subjects) + 1):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
    AUC = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw = 2, color = np.random.rand(3,), linestyle = line_styles[i % len(line_styles)], label = f'ROC curve for {target_names[i]} with AUC = {round(AUC, 5)}')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'best')
plt.show()


joblib.dump(model, 'data/models/test3.joblib')
model.save_weights('data/models/keras model 1/')




