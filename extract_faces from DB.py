import cv2, numpy as np, os, face_recognition, utils
from tqdm import tqdm
from keras_facenet import FaceNet
input_paths = ['data/database collage/DB cam 1 version 5']
output_paths = ['data/database collage/test']
for input_path, output_path in zip(input_paths, output_paths):
    paths = os.listdir(input_path)
    paths = [os.path.join(input_path, path) for path in paths]
    filenames = [os.path.join(path, filename) for path in paths for filename in os.listdir(path)]
    
    face_number = 0
    for i in tqdm(range(len(filenames))):
        try:
            image = cv2.imread(filenames[i])
            faces_locations = face_recognition.face_locations(image, 1, 'cnn')
            for top, right, bottom, left in faces_locations:
                face = image[top:bottom, left:right]
                source = filenames[i].replace('/', '_').replace('\\', '_')
                cv2.imwrite(os.path.join(output_path, f'Image {face_number}  Source_{source}'), face)
                face_number += 1        
        except:
            continue        



