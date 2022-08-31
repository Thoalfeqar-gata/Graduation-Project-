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
from keras.applications.vgg16 import VGG16

samples =  745
path = 'data/modalities/'

size = (320, 320)

left_eyes = [os.path.join(path, x) for x in os.listdir(path) if 'left eye' in x]
right_eyes = [os.path.join(path, x) for x in os.listdir(path) if 'right eye' in x]
mouths = [os.path.join(path, x) for x in os.listdir(path) if 'mouth' in x]
noses = [os.path.join(path, x) for x in os.listdir(path) if 'nose' in x]
faces = [os.path.join('data/preprocessed', x) for x in os.listdir('data/preprocessed')]
negatives = [os.path.join('data/negative/img', x) for x in os.listdir('data/negative/img')]
