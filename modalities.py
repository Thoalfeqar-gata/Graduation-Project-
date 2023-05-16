import cv2, os, numpy as np, utils
import mediapipe, face_recognition
from keras_vggface import VGGFace
from tqdm import tqdm
from deepface import DeepFace
from cv2 import dnn_superres
from keras.applications import vgg16, vgg19, ResNet50
from descriptors.GIST import GIST
from skimage.feature import hog
from descriptors.LocalDescriptors import WeberPattern, LocalBinaryPattern
from descriptors.BOWDescriptors import SIFTBOWFeatures, SURFBOWFeatures
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import train_test_split
import fusion
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import label_binarize
from matplotlib import pyplot as plt
from keras.callbacks import EarlyStopping
from sklearn.metrics import f1_score, precision_score, recall_score


size = 50
param = {
    'orientationsPerScale' : np.array([8, 8, 8, 8]),
    'boundaryExtension' : 1,
    'numberBlocks' : [4, 4],
    'fc_prefilt' : 10
}
gist = GIST(param)

vgg16_ = vgg16.VGG16(False, input_shape = (size, size, 3))
for layer in vgg16_.layers:
    layer.trainable = False
vgg16_features = lambda images: np.array(vgg16_.predict(images, 16)).reshape(len(images), -1)

vgg19_ = vgg19.VGG19(False, input_shape = (size, size, 3))
for layer in vgg19_.layers:
    layer.trainable = False
vgg19_features = lambda images: np.array(vgg19_.predict(images, 16)).reshape(len(images), -1)

vggface = VGGFace(False, input_shape = (size, size, 3))
for layer in vggface.layers:
    layer.trainable = False
vggface_features = lambda images: np.array(vggface.predict(images, 16)).reshape(len(images), -1)

def deepface_features(images):
    features = []
    for i in tqdm(range(len(images))):
        f = DeepFace.represent(images[i], 'DeepFace', detector_backend = 'opencv', enforce_detection = False)
        features.append(f)
    return features
        
        
def hog_features(images):
    features = []
    for i in tqdm(range(len(images))):
        image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        f = hog(image, 10, (8, 8))
        features.append(f)
    
    return features

def gist_features(images):
    features = []
    print('Processing gist features...')
    for i in tqdm(range(len(images))):
        image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        f = gist._gist_extract(image)
        features.append(f)
    return features

def face_embeddings(images):
    features = []
    for i in tqdm(range(len(images))):
        embedding = np.array(face_recognition.face_encodings(images[i], model = 'large'))
        
        if embedding.shape == (0,):
            embedding = np.zeros((128,))
        else:
            embedding = embedding[0]
        
        features.append(embedding)
        
    return features

weber = WeberPattern((4, 4))
lbp = LocalBinaryPattern(20, 3, (7,7))

path = 'data/database collage/detections/DB unified/all faces with augmentation'
training_data = []
training_labels = []
class_names = [f'{i}' for i in range(len(os.listdir(path)))][0:42]
images = []


mesh_detector = mediapipe.solutions.face_mesh.FaceMesh(static_image_mode = True,
                                       max_num_faces = 1,
                                       refine_landmarks = True,
                                       min_detection_confidence = 0.5)

for dirname, dirnames, filenames in os.walk(path):
    if len(filenames) <= 0:
        continue
    if int(os.path.split(dirname)[1]) > 41:
        continue
    
    print(dirname)
    subject = os.path.split(dirname)[1]
    for i, filename in enumerate(filenames):
        image = cv2.imread(os.path.join(dirname, filename))
        image = cv2.pyrUp(cv2.pyrUp(image))
        points = utils.face_mesh_mp(image, mesh_detector)
        if i > 400:
            break
        if points is None:
            continue
        
        right_eye_points = points[68], points[51]
        left_eye_points = points[9], points[280]
        mouth_points = points[205], points[379]
        nose_points = points[65], points[436]
        
        right_eye = image[right_eye_points[0][1] : right_eye_points[1][1], right_eye_points[0][0] : right_eye_points[1][0]].copy()
        left_eye = image[left_eye_points[0][1] : left_eye_points[1][1], left_eye_points[0][0] : left_eye_points[1][0]].copy()
        mouth = image[mouth_points[0][1] : mouth_points[1][1], mouth_points[0][0] : mouth_points[1][0]].copy()
        nose = image[nose_points[0][1] : nose_points[1][1], nose_points[0][0] : nose_points[1][0]].copy()

        if (0 in right_eye.shape) or (0 in left_eye.shape) or (0 in mouth.shape) or (0 in nose.shape):
            continue
        
        s = int(subject)
        training_labels.append(s)
        
        right_eye = cv2.resize(right_eye, (size, size))
        left_eye = cv2.resize(left_eye, (size, size))
        mouth = cv2.resize(mouth, (size, size))
        nose = cv2.resize(nose, (size, size))
        
        images.append(right_eye)
        images.append(left_eye)
        images.append(mouth)
        images.append(nose)
        
images = np.array(images)
training_labels = np.array(training_labels)
fusion_obj = fusion.Fusion([], class_names = class_names)
feature_extraction_algorithms = {
    'SIFT' : SIFTBOWFeatures,
    'SURF' : SURFBOWFeatures,
    'GIST' : gist_features,
    'LBP' : lbp.compute,
    'Weber' : weber.compute,
    'HOG' : hog_features,
    'VGG16' : vgg16_features,
    'VGG19' : vgg19_features,
    'VGGFace' : vggface_features,
    'Face embeddings' : face_embeddings
}
classification_algorithms = ['SVM', 'Neural Network']


for classification_algorithm in classification_algorithms:
    for feature_extraction_algorithm in feature_extraction_algorithms.keys():
        training_data = np.array(feature_extraction_algorithms[feature_extraction_algorithm](images)).reshape(len(training_labels), -1)
        X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size = 0.25, train_size = 0.75, random_state = 250)
        
        if classification_algorithm == 'SVM':
            SVM = OneVsRestClassifier(SVC(probability = True, verbose = True), n_jobs = -1)
            SVM.fit(X_train, y_train)
            y_pred_proba = SVM.predict_proba(X_test)
            y_pred = np.argmax(y_pred_proba, axis = -1)
            y_bin = label_binarize(y_test, classes = np.unique(training_labels))
        elif classification_algorithm == 'Neural Network':
            layers = [
            Dense(192, 'relu'),
            Dense(256, 'relu'),
            Dense(128, 'relu'),
            Dense(len(class_names), 'softmax')
            ]
            NeuralNetwork = Sequential(layers)
            NeuralNetwork.compile('adam', 'sparse_categorical_crossentropy', ['accuracy'])
            callback = EarlyStopping(patience = 30, verbose = 1, restore_best_weights = True, monitor = 'val_accuracy')

            NeuralNetwork.fit(X_train, y_train, 32, 100, callbacks = [callback], validation_split = 0.1, shuffle = False, use_multiprocessing = True)
            y_pred_proba = NeuralNetwork.predict(X_test)
            y_pred = np.argmax(y_pred_proba, axis = -1)
            y_bin = label_binarize(y_test, classes = np.unique(training_labels))
        
        f1 = f1_score(y_test, y_pred, average = 'weighted')
        precision = precision_score(y_test, y_pred, average = 'weighted')
        recall = recall_score(y_test, y_pred, average = 'weighted')
        with open('data/results2/results.txt', 'a') as results:
            results.write(f'Results for {feature_extraction_algorithm} using {classification_algorithm} on modalities. f1 : {f1}, precision : {precision}, recall : {recall}\n')
            
        fusion_obj.ROC_curve(y_bin, y_pred_proba, separate_subjects = False, roc_title = f'ROC curve for {feature_extraction_algorithm} using {classification_algorithm} on modalities')
        fusion_obj.confusion_matrix(y_pred, y_test, matrix_title = f'Confusion matrix for {feature_extraction_algorithm} using {classification_algorithm} on modalities')