from sklearn.datasets import fetch_lfw_people
from keras_vggface import VGGFace
from keras.applications.vgg16 import VGG16
from sklearn.model_selection import train_test_split
from fusion import FeatureFusion, ScoreFusion
from keras.models import Sequential
from keras.layers import Input, Dense
from keras.optimizers import Adam
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from sklearn.metrics import classification_report, roc_curve, auc
from tqdm import tqdm
from skimage.feature import hog
from descriptors.LocalDescriptors import WeberPattern, LocalBinaryPattern
import face_recognition
import cv2, numpy as np, os

image_paths = []
subjects = []
size = 100

images_per_subject = 45
subjects_count = 20

i = 0
for dir, dirnames, filenames in os.walk('data/lfw/lfw augmented'):

    if len(filenames) == 0:
        continue
    
    person_images = []
    for filename in filenames:
        img_path = os.path.join(dir, filename)
        person_images.append(img_path)
        
        if len(person_images) >= images_per_subject:
            break
     
    image_paths.append(person_images)
    subjects.append(os.path.split(dir)[1])

    i = i + 1
    if subjects_count is not None:
        if i >= subjects_count:
            break
    


dnn = VGGFace(False, input_shape = (size, size, 3))

for layer in dnn.layers:
    layer.trainable = False

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
lbp = LocalBinaryPattern(32, 8, (5,5))

DNN = lambda images: np.array(dnn.predict(images, 16)).reshape(len(images), -1)


fusion = FeatureFusion([
   DNN,
   face_embeddings
],
    subjects)

fusion.extract_features(image_paths, image_size = (size, size))
fusion.train_svm(flip = False, roc_title = 'Augmented database')
