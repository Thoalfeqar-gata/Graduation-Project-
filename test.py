import cv2, joblib, numpy as np, time, face_recognition, os, multiprocessing as mp
from tqdm import tqdm
from keras.applications.vgg16 import VGG16
from keras_vggface import VGGFace
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from descriptors.fusion import FeatureFusion, ScoreFusion
from descriptors.LocalDescriptors import WeberPattern, LocalBinaryPattern
from skimage.feature import hog
from descriptors.BOWDescriptors import SIFTBOWFeatures, SURFBOWFeatures
from descriptors.GIST import GIST

size = 70
weber = WeberPattern((2, 2))
lbp = LocalBinaryPattern(32, 3, (6,6))

    
dnn = VGGFace(include_top = False, input_shape = (size, size, 3))
for layer in dnn.layers:
    layer.trainable = False
    
DNN = lambda images: np.array(dnn.predict(images, 16, use_multiprocessing = True)).reshape(len(images), -1)

def hog_features(images):
    print('Processing hog features...')
    features = []
    for i in tqdm(range(len(images))):
        image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        f = hog(image, 10, (8, 8))
        features.append(f)
    
    return features


def face_embeddings(images):
    features = []
    for i in tqdm(range(len(images))):
        embedding = np.array(face_recognition.face_encodings(images[i], num_jitters = 1, model = 'large'))
        
        
        if embedding.shape == (0,):
            embedding = np.zeros((128,))
        else:
            embedding = embedding[0]
        
        features.append(embedding)

            
    return features


number_of_subjects = 37
authorised_subjects = [i for i in range(1, number_of_subjects + 1)]

path = os.path.join('data', 'database collage', 'detections', 'all faces with augmentation')
subjects = [i for i in range(1, number_of_subjects + 1)]
print(subjects)
print(len(subjects))

images_list = []
unauthorised_list = []

for subject in subjects:
    images_folder = os.path.join(path, str(subject))
    image_paths = [os.path.join(images_folder, image) for image in os.listdir(images_folder)]  
    
    if subject in authorised_subjects:
        images_list.append(image_paths)
    else:
        unauthorised_list.extend(image_paths)

if len(unauthorised_list) > 0:
    images_list.append(unauthorised_list)


class_names = [f'Subject: {i}' for i in authorised_subjects]


fusion = FeatureFusion([
    weber.compute,
    hog_features
], 
   class_names, image_size = (size, size))

fusion.extract_features(images_list)
del images_list
fusion.train_svm()