import cv2, numpy as np, os, face_recognition, mediapipe as mp, dlib
from tqdm import tqdm
detector = dlib.get_frontal_face_detector()
shape_predictor = dlib.shape_predictor('data/dlib data/shape_predictor_68_face_landmarks.dat')

dataset_path = 'data/lfw/lfw augmented'
output_dataset_path = 'data/lfw/lfw cropped'
subjects = os.listdir(dataset_path)

for index in tqdm(range(len(subjects))):
    
    filenames = os.listdir(os.path.join(dataset_path, subjects[index]))

    for i, filename in enumerate(filenames):
        image = cv2.imread(os.path.join(dataset_path, subjects[index], filename))
        

        try:
            boxes = detector(image, 1)
            d = boxes[0]
            shape = shape_predictor(image, d)
            points = shape.parts()
        
        
            padding_eyes = 13
            padding_mouth = 7
            padding_nose = 5
            left_eye = image[points[38].y - padding_eyes : points[40].y + padding_eyes, points[36].x - padding_eyes : points[39].x + padding_eyes]
            right_eye = image[points[44].y - padding_eyes : points[46].y + padding_eyes, points[42].x - padding_eyes : points[45].x + padding_eyes]
            mouth = image[points[50].y - padding_mouth : points[57].y + padding_mouth, points[48].x - padding_mouth : points[54].x + padding_mouth]
            nose = image[points[27].y - padding_nose : points[33].y + padding_nose, points[31].x - padding_nose : points[35].x + padding_nose]
        
        
            output_path = os.path.join('data/lfw/lfw modalities', subjects[index])
            cv2.imwrite(os.path.join(output_path, f'left_eye {i}.jpg'), left_eye)
            cv2.imwrite(os.path.join(output_path, f'right_eye {i}.jpg'), right_eye)
            cv2.imwrite(os.path.join(output_path, f'mouth {i}.jpg'), mouth)
            cv2.imwrite(os.path.join(output_path, f'nose {i}.jpg'), nose)
        except:
            continue


       
        
        
        
        
        
        

    