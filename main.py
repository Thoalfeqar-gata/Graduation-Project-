import cv2, os, numpy as np, random, time, pickle
from tqdm import tqdm
from skimage.feature import hog, local_binary_pattern
from skimage import data
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from descriptors.LocalDescriptors import WeberPattern, LocalBinaryPattern
from matplotlib import pyplot as plt
from sklearn.preprocessing import label_binarize
from keras.applications.resnet import preprocess_input, ResNet50
from keras.utils import to_categorical
from descriptors.GIST import GIST

samples =  745
path = 'data/modalities/'

size = 224

left_eyes = [os.path.join(path, x) for x in os.listdir(path) if 'left eye' in x]
right_eyes = [os.path.join(path, x) for x in os.listdir(path) if 'right eye' in x]
mouths = [os.path.join(path, x) for x in os.listdir(path) if 'mouth' in x]
noses = [os.path.join(path, x) for x in os.listdir(path) if 'nose' in x]
faces = [os.path.join('data/preprocessed', x) for x in os.listdir('data/preprocessed')]
negatives = [os.path.join('data/negative/img', x) for x in os.listdir('data/negative/img')]

train_data = []
train_labels = []

param = {
        "orientationsPerScale":np.array([8,8]),
         "numberBlocks":[10,10],
        "fc_prefilt":10,
        "boundaryExtension": 10
        
}

gist = GIST(param)

for i in tqdm(range(samples)):
    left_eye = gist._gist_extract(cv2.resize(cv2.imread(left_eyes[i], cv2.IMREAD_GRAYSCALE), (size, size)))
    right_eye = gist._gist_extract(cv2.resize(cv2.imread(right_eyes[i], cv2.IMREAD_GRAYSCALE), (size, size)))
    mouth = gist._gist_extract(cv2.resize(cv2.imread(mouths[i], cv2.IMREAD_GRAYSCALE), (size, size)))
    nose = gist._gist_extract(cv2.resize(cv2.imread(noses[i], cv2.IMREAD_GRAYSCALE), (size, size)))

    train_data.append(left_eye)
    train_data.append(right_eye)
    train_data.append(mouth)
    train_data.append(nose)
    
    train_labels.append(0)
    train_labels.append(1)
    train_labels.append(2)
    train_labels.append(3)
    
    # face = gist._gist_extract(cv2.resize(cv2.imread(faces[i], cv2.IMREAD_GRAYSCALE), (size, size)))
    # negative = gist._gist_extract(cv2.resize(cv2.imread(negatives[i], cv2.IMREAD_GRAYSCALE), (size, size)))
    
    # train_data.append(face)
    # train_data.append(negative)
    # train_labels.append(0)
    # train_labels.append(1)
    
    
# train_images = np.array(train_images)
# # train_labels = label_binarize(np.array(train_labels), classes = [0, 1])
# train_images = preprocess_input(train_images)

# cnn_model = ResNet50(include_top = False, weights = 'imagenet', input_shape = (size, size, 3))

# for layer in cnn_model.layers:
#     layer.trainable = False
    
# print(cnn_model.summary())

# train_data = cnn_model.predict(train_images)
# train_data = train_data.reshape(train_data.shape[0], -1)
train_labels = label_binarize(train_labels, classes = [0, 1, 2, 3])
X_train, X_test, y_train, y_test = train_test_split(train_data, train_labels, test_size = 0.25, train_size = 0.75)
mlp = MLPClassifier((196, 256, 128), max_iter = 10000)
mlp.fit(X_train, y_train)
y_true = np.array(y_test)


y_score = mlp.predict_proba(X_test)
y_pred = mlp.predict(X_test)

# print(classification_report(y_true, y_pred, target_names = ['left_eye', 'right_eye', 'mouth', 'nose']))
modalities = ['left_eye', 'right_eye', 'mouth', 'nose']
line_styles = [':', '-', '--', '-.']
colors = ['darkorange', 'forestgreen', 'aqua', 'salmon']

for i in range(len(modalities)):
    fpr, tpr, _ = roc_curve(y_true[:, i], y_score[:, i])
    AUC = round(auc(fpr, tpr), 5)
    
    plt.plot(fpr, tpr, lw = 2, color = colors[i], linestyle = line_styles[i], label = f'ROC curve for {modalities[i]} with AUC = {AUC}')


plt.title(f'ROC curve for face detection with GIST')
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.legend(loc = 'lower right')
plt.show()