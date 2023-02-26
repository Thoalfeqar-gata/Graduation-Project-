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
subjects = []
total_images = 0
for dirname, dirnames, filenames in os.walk(path):
    if len(filenames) <= 0:
        continue
    subject_images = [os.path.join(dirname, filename) for filename in filenames]
    faces_paths.append(subject_images)
    total_images += len(subject_images)
    subjects.append(f'S{os.path.split(dirname)[1]}')

    
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
vggface_features = lambda images: np.array(vggface.predict(images, 16)).reshape(len(images), -1)

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
        embedding = np.array(face_recognition.face_encodings(images[i], model = 'large'))
        
        if embedding.shape == (0,):
            embedding = np.zeros((128,))
        else:
            embedding = embedding[0]
        
        features.append(embedding)
        
    return features

weber = WeberPattern((4, 4))
lbp = LocalBinaryPattern(20, 3, (7,7))
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

fusion_obj = FeatureFusion([
    hog_features
], subjects)
fusion_obj.extract_features(faces_paths, image_size = (size, size))
model = fusion_obj.train(250, 32, patience = 25, separate_subjects = False, roc_title = 'Delete me', matrix_title = 'Delete me matrix', results_title = 'Delete me')
model.save('data/models/hog with feature fusion')
