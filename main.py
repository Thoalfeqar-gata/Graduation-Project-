import cv2, os, numpy as np, random, time, pickle
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
from sklearn.svm import SVC
from descriptors.LocalDescriptors import WeberPattern, LocalBinaryPattern
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.utils import to_categorical
from descriptors.GIST import GIST

size = 224
path = 'data/modalities'
people_count = 1889
left_eyes = [os.path.join(path, 'left eye', image) for image in os.listdir(os.path.join(path, 'left eye')) if image.endswith('.jpg')][:people_count]
right_eyes = [os.path.join(path, 'right eye', image) for image in os.listdir(os.path.join(path, 'right eye')) if image.endswith('.jpg')][:people_count]
mouths = [os.path.join(path, 'mouth', image) for image in os.listdir(os.path.join(path, 'mouth')) if image.endswith('.jpg')][:people_count]
noses = [os.path.join(path, 'nose', image) for image in os.listdir(os.path.join(path, 'nose')) if image.endswith('.jpg')][:people_count]
faces = [os.path.join('data/preprocessed', x) for x in os.listdir('data/preprocessed')]
negatives = [os.path.join('data/negative/img', x) for x in os.listdir('data/negative/img')]

dnn = VGG16(include_top = False, weights = 'imagenet', input_shape = (size, size, 3))
for layer in dnn.layers:
    layer.trainable = False

param = {
        "orientationsPerScale":np.array([5,5]),
         "numberBlocks":[5,5],
        "fc_prefilt":10,
        "boundaryExtension": 10 
}

weber = WeberPattern(grid_shape = (7, 7))
lbp = LocalBinaryPattern(num_points = 32, radius = 8, grid_shape = (7, 7))
gist = GIST(param)

def hog_features(images):
    features = []
    print('processing hog features...')
    for i in tqdm(range(len(images))):
        img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        f = hog(img)
        features.append(f)
    return features

def gist_features(images):
    features = []
    print('processing GIST features...')
    for i in tqdm(range(len(images))):
        img = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        f = gist._gist_extract(img)
        features.append(f)
    return features

#all the samples are equal in amount
images_number = len(left_eyes)
left_eye_images = []
right_eye_images = []
mouth_images = []
nose_images = []
face_images = []
negative_images = []

print('loading images...')
for i in tqdm(range(images_number)):
    left_eye = cv2.resize(cv2.imread(left_eyes[i]), (size, size))
    right_eye = cv2.resize(cv2.imread(right_eyes[i]), (size, size))
    mouth= cv2.resize(cv2.imread(mouths[i]), (size, size))
    nose = cv2.resize(cv2.imread(noses[i]), (size, size))
    face = cv2.resize(cv2.imread(faces[i]), (size, size))
    negative = cv2.resize(cv2.imread(negatives[i % 1570]), (size, size))
    
    left_eye_images.append(left_eye)
    right_eye_images.append(right_eye)
    mouth_images.append(mouth)
    nose_images.append(nose)
    face_images.append(face)
    negative_images.append(negative)
    
left_eye_images = np.array(left_eye_images)
right_eye_images = np.array(right_eye_images)
mouth_images = np.array(mouth_images)
nose_images = np.array(nose_images)


score_fusion = fusion.FeatureFusion(
    algorithms = {
        'vgg16' : dnn.predict,
        'GIST' :  gist_features,      
        'weber' : weber.compute,
        'hog' : hog_features
    },
    class_names = [
       'left eye',
       'right eye',
       'mouth', 
       'nose'                                     
    ]
)

score_fusion.extract_features([
    left_eye_images,
    right_eye_images,
    mouth_images,
    nose_images
])

del left_eye_images
del right_eye_images
del mouth_images
del nose_images

score_fusion.train(epochs = 100)

