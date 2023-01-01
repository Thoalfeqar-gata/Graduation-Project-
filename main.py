import cv2, os, numpy as np, random, time, pickle, joblib, random, face_recognition
import utils, tensorflow as tf, keras, fusion
from descriptors.BOWDescriptors import SIFTBOWFeatures, SURFBOWFeatures
from keras.layers import Dense, Input
from keras.models import Sequential
from keras.optimizers import Adam
from keras.metrics import categorical_crossentropy
from tqdm import tqdm
from keras_vggface import VGGFace
from deepface import DeepFace
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
from fusion import FeatureFusion, ScoreFusion

path = 'data/database collage/detections/DB unified/all faces with augmentation'
size = 100

image_paths = []
subjects = [f'S{i}' for i in range(len(os.listdir(path)))]
print(subjects)
for dirname, dirnames, filenames in os.walk(path):
    if len(filenames) <= 0:
        continue
    
    subject_images = [os.path.join(dirname, filename) for filename in filenames]
    image_paths.append(subject_images)


    
dnn = VGGFace(False, input_shape = (size, size, 3))
for layer in dnn.layers:
    layer.trainable = False
DNN = lambda images: np.array(dnn.predict(images, 16)).reshape(len(images), -1)

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

weber = WeberPattern((2, 2))
lbp = LocalBinaryPattern(20, 3, (7,7))

fusion = FeatureFusion([
   DNN
],
    subjects)

fusion.extract_features(image_paths, image_size = (size, size))
model = fusion.train_svm(flip = False, roc_title = 'Augmented database')
# pickle.dump(model, open('data/models/database collage svm with vggface/model.sav', 'wb'))