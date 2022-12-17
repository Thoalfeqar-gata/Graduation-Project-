import cv2, numpy as np, os, face_recognition, mediapipe as mp, dlib
from tqdm import tqdm
import utils

path = 'data/database collage/DB version 2'
output_path = 'data/database collage/detections/DB version 2/All faces'
utils.extract_faces_from_database(path, output_path)