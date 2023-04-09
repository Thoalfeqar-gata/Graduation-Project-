import cv2, numpy as np, os, face_recognition, utils, time
from tqdm import tqdm
from keras_facenet import FaceNet
input_paths = ['data/database collage/DB cam 1 version 5']
output_paths = ['data/database collage/test']
def extract_faces_from_images(input_paths, output_paths):
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

def extract_faces_from_videos(input_path, output_path, video_size = (960, 540)):
    
    for filename in os.listdir(input_path):
        cap = cv2.VideoCapture(os.path.join(input_path, filename))
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        print(filename)
        if filename[0] != '4':
            continue
        for _ in tqdm(range(frame_count)):
            ret, frame = cap.read()
            temp = cv2.resize(frame, video_size)
            locations = face_recognition.face_locations(temp, 1, 'hog')
            for top, right, bottom, left in locations:
                top = int(top/video_size[1] * frame_height)
                bottom = int(bottom/video_size[1] * frame_height)
                left = int(left/video_size[0] * frame_width)
                right = int(right/video_size[0] * frame_width)
                
                face = frame[top : bottom, left : right]
                cv2.imwrite(os.path.join(output_path, filename[0], f'{time.time()}.jpg'), face)

extract_faces_from_videos('data/database collage/DB cam 4 version 1 (friends videos with phone)', 'data/database collage/detections/DB version 6 (video)')


