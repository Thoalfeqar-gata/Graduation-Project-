import cv2, joblib, numpy as np, time, face_recognition, os
from tqdm import tqdm
from keras.applications.vgg16 import VGG16
from keras.models import Sequential
from keras.layers import Dense, Input
from keras.optimizers import Adam
from fusion import FeatureFusion, ScoreFusion
from descriptors.LocalDescriptors import WeberPattern, LocalBinaryPattern
from skimage.feature import hog
from descriptors.BOWDescriptors import SIFTBOWFeatures, SURFBOWFeatures

def hog_features(images):
    features = []
    for i in tqdm(range(len(images))):
        image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
        f = hog(image, 10, (8, 8))
        features.append(f)
    
    return features

size = 224
weber = WeberPattern((6, 6))
lbp = LocalBinaryPattern(8, 1, (6,6))

dnn = VGG16(include_top = False, input_shape = (size, size, 3))
for layer in dnn.layers:
    layer.trainable = False

number_of_subjects = 37
authorised_subjects = [3, 5, 7, 11, 37]

path = os.path.join('data', 'database collage', 'detections', 'all faces with augmentation')
subjects = [i for i in range(1, number_of_subjects + 1)]
print(subjects)


images_list = []
unauthorised_list = []
samples_count = 200
for subject in subjects:
    images_folder = os.path.join(path, str(subject))
    image_paths = [os.path.join(images_folder, image) for image in os.listdir(images_folder)]
    
    images = []
    for image_path in image_paths:
        image = cv2.resize(cv2.imread(image_path), (size, size))
        images.append(image)
        
        if(len(images) >= samples_count):
            break
    
    if subject in authorised_subjects:
        images_list.append(images)
    else:
        unauthorised_list.extend(images)

images_list.append(unauthorised_list)



        
class_names = [f'Subject: {i}' for i in authorised_subjects]
class_names.append('Subject: 0')

fusion = FeatureFusion([
    SIFTBOWFeatures,
    SURFBOWFeatures

], 
   class_names, (1024, 512, 384))

fusion.extract_features(images_list)
fusion.train(300)
    