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
from deepface import DeepFace
import tensorflow as tf, dlib, pickle

image_paths = []
subjects = []
size = 160

images_per_subject = 45
subjects_count = None

i = 0
total_image_count = 0
for dir, dirnames, filenames in os.walk('data/lfw/lfw augmented'):

    if len(filenames) == 0:
        continue
    
    person_images = []
    for filename in filenames:
        img_path = os.path.join(dir, filename)
        person_images.append(img_path)
        
        if len(person_images) >= images_per_subject:
            break
     
    total_image_count += len(person_images)
    image_paths.append(person_images)
    subjects.append(f'{i}')

    i = i + 1
    if subjects_count is not None:
        if i >= subjects_count:
            break

print('The amount of images: ', total_image_count)
print('The number of subjects: ', i)

dnn = VGGFace(False, input_shape = (size, size, 3))

for layer in dnn.layers:
    layer.trainable = False

def deepface_features(images):
    features = []
    for i in tqdm(range(len(images))):
        f = DeepFace.represent(images[i], 'Facenet', detector_backend = 'dlib', enforce_detection = False)
        features.append(f)
    return features

shape_predictor = dlib.shape_predictor('data/dlib data/shape_predictor_5_face_landmarks.dat')
def preprocess_faces(image, detections, size = 160, normalize = True):
    global shape_predictor
    
    if len(detections) <= 0:
        detections = [(0, 0, image.shape[1], image.shape[0])]

    dets = []
    for x, y, w, h in detections:
        dets.append(dlib.rectangle(left = x, top = y, right = x+w, bottom = y+h))
    
    faces = dlib.full_object_detections()
    for det in dets:
        faces.append(shape_predictor(image, det))
    
    if normalize:
        faces = [face[:, :, ::-1] / 255.0 for face in dlib.get_face_chips(image, faces, size = size, padding = 0.075)]
    else:
        faces = [face[:, :, ::-1] for face in dlib.get_face_chips(image, faces, size = size, padding = 0.075)]
    return faces
    
feature_extractor = tf.lite.Interpreter(model_path = 'Raspberry Pi/models/facenet optimized/model.tflite', num_threads = 10)
feature_extractor.allocate_tensors()
feature_extractor_input_details = feature_extractor.get_input_details()
feature_extractor_output_details = feature_extractor.get_output_details()

def light_facenet(images):
    features = []
    for i in tqdm(range(len(images))):
        image = preprocess_faces(images[i], [], size = size)[0]
        image = np.expand_dims(image, 0)
        feature_extractor.set_tensor(feature_extractor_input_details[0]['index'], image.astype(np.float32))
        feature_extractor.invoke()
        feature = feature_extractor.get_tensor(feature_extractor_output_details[0]['index'])[0]
        features.append(feature)
    
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
lbp = LocalBinaryPattern(32, 8, (5,5))

DNN = lambda images: np.array(dnn.predict(images, 16)).reshape(len(images), -1)


fusion = ScoreFusion([
   deepface_features,
   face_embeddings
],
    subjects, [2, 3])

fusion.extract_features(image_paths, image_size = (size, size))
svm_models = fusion.train_svm(separate_subjects = False, roc_title = 'ROC curve for lfw (deepface + face_embeddings) using svm', results_title = 'lfw (deepface + face_embeddings) using svm', matrix_title = 'CM for lfw (deepface + face_embeddings) using svm')
NN_models = fusion.train(patience = 50, epochs = 250, batch_size = 32, separate_subjects = False, roc_title = 'ROC curve for lfw (deepface + face_embeddings) using NN', results_title = 'lfw (deepface + face_embeddings) using NN', matrix_title = 'CM for lfw (deepface + face_embeddings) using NN')


with open('data/extracted features/lfw deepface and face_embeddings score fusion.txt', 'wb') as file:
    pickle.dump((fusion.training_data, fusion.training_labels), file)
    
with open('data/models/lfw (deepface + face_embeddings) svm.txt', 'wb') as file:
    pickle.dump(svm_models, file)

with open('data/models/lfw (deepface + face_embeddings) NN.txt', 'wb') as file:
    pickle.dump(NN_models, file)

