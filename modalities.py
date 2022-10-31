import cv2, os, numpy as np, random, time, pickle, joblib, dlib
import utils, tensorflow as tf, keras, fusion, mediapipe
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
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from descriptors.LocalDescriptors import WeberPattern, LocalBinaryPattern
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from keras.applications.vgg16 import preprocess_input, VGG16
from keras.utils import to_categorical
from descriptors.GIST import GIST

mesh_detector = mediapipe.solutions.face_mesh.FaceMesh(static_image_mode = True,
                                       max_num_faces = 1,
                                       refine_landmarks = True,
                                       min_detection_confidence = 0.5)



number_of_subjects = 15
authorised_subjects = [2, 10, 20, 22, 37]
path = os.path.join('data', 'database collage', 'detections', 'all faces with augmentation')
subjects = [str(i) for i in range(number_of_subjects)]
subjects.extend(['20', '22', '37'])
print(subjects)
size = 96


dnn = VGG16(include_top = False, weights = 'imagenet', input_shape = (size, size, 3))
for layer in dnn.layers:
    layer.trainable = False

training_data = []
training_labels = []

def hog_features(images):
    features = []
    for i in tqdm(range(len(images))):
        f = hog(images[i])
        features.append(f)
    
    return np.array(features)

i = 1
for subject in subjects:
    images_folder = os.path.join(path, subject)
    image_paths = [os.path.join(images_folder, image) for image in os.listdir(images_folder)]
    
    
    face_images = []
    modality_images = []
    for image in image_paths:
        img = cv2.imread(image)
        modalities_start = image.find('R')
        
        if not subject == '37':
            right_eye_visible = bool(int(image[modalities_start + 2]))
            left_eye_visible = bool(int(image[modalities_start + 6]))
            nose_visible = bool(int(image[modalities_start + 10]))
            mouth_visible = bool(int(image[modalities_start + 14]))
        else:
            right_eye_visible = True
            left_eye_visible = True
            nose_visible = True
            mouth_visible = True
        
        try:
            points = np.array(utils.face_mesh_mp(img, mesh_detector))[:, :2].astype(np.uint8)
        except:
            continue
        
        right_eye_points = points[68], points[51]
        left_eye_points = points[9], points[280]
        mouth_points = points[205], points[379]
        nose_points = points[65], points[436]
        
        right_eye = img[right_eye_points[0][1] : right_eye_points[1][1], right_eye_points[0][0] : right_eye_points[1][0]].copy()
        left_eye = img[left_eye_points[0][1] : left_eye_points[1][1], left_eye_points[0][0] : left_eye_points[1][0]].copy()
        mouth = img[mouth_points[0][1] : mouth_points[1][1], mouth_points[0][0] : mouth_points[1][0]].copy()
        nose = img[nose_points[0][1] : nose_points[1][1], nose_points[0][0] : nose_points[1][0]].copy()
        
        
        if not right_eye_visible:
            right_eye = left_eye
            if not mouth_visible:
                mouth = left_eye
                
        if not left_eye_visible:
            left_eye = right_eye
            if not mouth_visible:
                mouth = right_eye
        
        if not mouth_visible:
            mouth = left_eye
        
        
        try:
            right_eye = cv2.resize(right_eye, (size, size))
            left_eye = cv2.resize(left_eye, (size, size))
            mouth = cv2.resize(mouth, (size, size))
            nose = cv2.resize(nose, (size, size))
        except:
            continue
        
        img = cv2.resize(img, (224, 224))
        face_images.append(img)
        modality_images.append(right_eye)
        modality_images.append(left_eye)
        modality_images.append(mouth)
        modality_images.append(nose)
    
    modality_images = np.array(modality_images)
    face_images = np.array(face_images)
    modality_features = np.array(dnn.predict(modality_images)).reshape(len(modality_images) // 4, -1)
    face_features = np.array(dnn.predict(face_images)).reshape(len(face_images), -1)
    features = np.concatenate((face_features, modality_features), axis = 1)
    
    training_data.extend(features)
    
    if int(subject) in authorised_subjects:
        training_labels.extend([i] * len(features))
        i = i + 1
    else:
        training_labels.extend([0] * len(features))

training_data = np.array(training_data)
training_labels = np.array(training_labels)
X_train, X_test, y_train, y_test = train_test_split(training_data, training_labels, test_size = 0.25, train_size = 0.75, shuffle = True, random_state = 250)
print(np.unique(training_labels))

model = Sequential([
    Input(shape = (training_data.shape[1])), 
    Dense(192, 'relu'),
    Dense(256, 'relu'),
    Dense(128, 'relu'),
    Dense(len(authorised_subjects) + 1, 'softmax')
])

model.compile(Adam(), 'sparse_categorical_crossentropy', ['accuracy'])
model.fit(X_train, y_train, shuffle = False, use_multiprocessing = True, epochs = 75, verbose = 1)

y_pred_prob = model.predict(X_test) 
y_bin = label_binarize(y_test, classes = list(range(len(authorised_subjects) + 1)))

line_styles = [':', '-', '--', '-.']
target_names = ['Subject: Unknown']
target_names.extend([f'Subject: {subject}' for subject in authorised_subjects])
print(classification_report(y_test, np.argmax(y_pred_prob, -1), target_names = target_names))

for i in range(len(authorised_subjects) + 1):
    fpr, tpr, _ = roc_curve(y_bin[:, i], y_pred_prob[:, i])
    AUC = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw = 2, color = np.random.rand(3,), linestyle = line_styles[i % len(line_styles)], label = f'ROC curve for {target_names[i]} with AUC = {round(AUC, 5)}')

plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'best')
plt.show()



    

        
    
        
        
        
        
    
 




