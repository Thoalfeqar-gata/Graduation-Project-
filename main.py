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
from matplotlib import pyplot as plt
from deepface import DeepFace
import pickle
from keras_facenet import FaceNet

param = {
    'orientationsPerScale' : np.array([8, 8, 8, 8]),
    'boundaryExtension' : 1,
    'numberBlocks' : [4, 4],
    'fc_prefilt' : 10
}
gist = GIST(param)
path = 'data/database collage/detections/DB unified of friends/DB with augmentation'
size = 180
images_per_subject = 1500
faces_paths = []
subjects = []
total_images = 0
for dirname, dirnames, filenames in os.walk(path):
    if len(filenames) <= 0:
        continue
    subject_images = [os.path.join(dirname, filename) for filename in filenames]
    faces_paths.append(subject_images)
    total_images += len(subject_images)
    subjects.append(f'{os.path.split(dirname)[1]}')

    
resnet50 = ResNet50(False, input_shape = (size, size, 3))
for layer in resnet50.layers:
    layer.trainable = False
resnet50_features = lambda images: np.array(resnet50.predict(images, 16)).reshape(len(images), -1)

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
vggface_features = lambda images: np.array(vggface.predict(np.array(images), 16)).reshape(len(images), -1)

deepface_model = DeepFace.build_model('Facenet')
def deepface_features(images):
    features = []
    for i in tqdm(range(len(images))):
        f = DeepFace.represent(images[i], model = deepface_model, detector_backend = 'opencv', enforce_detection = False)
        features.append(f)
    return features

def facenet(images):
    facenet_model = FaceNet()
    faces = []
    for i in tqdm(range(len(images))):
        try:
            top, right, bottom, left = face_recognition.face_locations(images[i], 1, model = 'cnn')[0]
            face_image = images[i][top:bottom, left:right]
            faces.append(face_image)
        except:
            faces.append(images[i])
            
    return np.array(facenet_model.embeddings(faces))
    
        
        
def hog_features(images):
    features = []
    for i in tqdm(range(len(images))):
        image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        f = hog(image, 10, (8, 8), (2, 2))
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
        embedding = np.array(face_recognition.face_encodings(images[i], num_jitters = 3, model = 'large'))
        
        if embedding.shape == (0,):
            print('hi')
            embedding = np.zeros((128,))
        else:
            embedding = embedding[0]
        
        features.append(embedding)
        
    return np.array(features)

weber = WeberPattern((4, 4))
lbp = LocalBinaryPattern(16, 2, (3,3))
# feature_extraction_algorithms = {
#     'SIFT' : SIFTBOWFeatures,
#     'SURF' : SURFBOWFeatures,
#     'GIST' : gist_features,
#     'LBP' : lbp.compute,
#     'Weber' : weber.compute,
#     'HOG' : hog_features,
#     'VGG16' : vgg16_features,
#     'VGG19' : vgg19_features,
#     'VGGFace' : vggface_features,
#     'Face embeddings' : face_embeddings
# }
# classification_algorithms = ['SVM', 'Neural Network']

# for classification_algorithm in classification_algorithms:
#     for feature_extraction_algorithm in feature_extraction_algorithms.keys():
#         fusion = FeatureFusion([
#             feature_extraction_algorithms[feature_extraction_algorithm]
#         ], subjects)
        
#         if feature_extraction_algorithm in ['SIFT', 'SURF']:
#             fusion.extract_features(faces_paths, batch_size = total_images, image_size = (size, size))
#         else:
#             fusion.extract_features(faces_paths, image_size = (size, size))

#         if classification_algorithm == 'SVM':
#             fusion.train_svm(separate_subjects = False, roc_title = f'ROC curve for {feature_extraction_algorithm} using {classification_algorithm} on faces.', matrix_title = f'Confusion matrix for {feature_extraction_algorithm} using {classification_algorithm} on faces', results_title = f'results for {feature_extraction_algorithm} using {classification_algorithm} on faces')
#         else:
#             fusion.train(100, 16, patience = 30, model_layer_sizes = (256, 384, 192), separate_subjects = False, roc_title = f'ROC curve for {feature_extraction_algorithm} using {classification_algorithm} on faces.', matrix_title = f'Confusion matrix for {feature_extraction_algorithm} using {classification_algorithm} on faces', results_title = f'results for {feature_extraction_algorithm} using {classification_algorithm} on faces')

with open('data/models/svm using dlib face embeddings/info.txt', 'w') as file:
    file.write(f'face_size = {(size, size)}')
fusion_obj = FeatureFusion([
    deepface_features
], subjects)
fusion_obj.extract_features(faces_paths, image_size = None)
model = fusion_obj.train_svm(separate_subjects = False, roc_title = 'ROC curve of face embeddings on friends database', matrix_title = 'Confusion matrix of face embeddings on friends database', results_title = 'face embeddings on friends database')
file = open('data/models/svm using dlib face embeddings/model.pickle', 'wb')
pickle.dump(model, file)