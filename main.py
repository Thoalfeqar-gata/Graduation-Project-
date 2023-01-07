import cv2, os, numpy as np, random, face_recognition
from descriptors.BOWDescriptors import SIFTBOWFeatures, SURFBOWFeatures
from tqdm import tqdm
from keras_vggface import VGGFace
from keras.applications import vgg16, vgg19, ResNet50
from deepface import DeepFace
from skimage.feature import hog
from descriptors.LocalDescriptors import WeberPattern, LocalBinaryPattern
from descriptors.GIST import GIST
from fusion import FeatureFusion, ScoreFusion

param = {
    'orientationsPerScale' : np.array([8, 8, 8, 8]),
    'boundaryExtension' : 1,
    'numberBlocks' : [4, 4],
    'fc_prefilt' : 10
}
gist = GIST(param)
path = 'data/database collage/detections/DB unified/all faces with augmentation'
size = 100

faces_paths = []
subjects = [f'S{i}' for i in range(len(os.listdir(path)))]
print(subjects)
total_images = 0
for dirname, dirnames, filenames in os.walk(path):
    if len(filenames) <= 0:
        continue
    subject_images = [os.path.join(dirname, filename) for filename in filenames]
    faces_paths.append(subject_images)
    total_images += len(subject_images)

    
dnn = ResNet50(False, input_shape = (size, size, 3))
for layer in dnn.layers:
    layer.trainable = False
DNN = lambda images: np.array(dnn.predict(images, 16)).reshape(len(images), -1)

def deepface_features(images):
    features = []
    for i in tqdm(range(len(images))):
        f = DeepFace.represent(images[i], 'Facenet', detector_backend = 'dlib', enforce_detection = False)
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

fusion = FeatureFusion([
    DNN
],
    subjects)
print(total_images)
fusion.extract_features(faces_paths,batch_size = total_images, image_size = (size, size))
model = fusion.train(100, 32, patience = 30, flip = False, roc_title = 'ResNet50')
