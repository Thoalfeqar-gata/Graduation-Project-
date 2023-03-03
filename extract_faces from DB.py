import cv2, numpy as np, os, face_recognition, utils
from tqdm import tqdm

# path = 'data/database collage/DB cam 3 version 1 (friends)'
# output_path = 'data/database collage/detections/DB unified/all faces'
# starting_subject_number = 43
# subjects = os.listdir(path)

# for i in tqdm(range(len(subjects))):
#     subject_path = os.path.join(path, subjects[i])
#     output_subject_path = os.path.join(output_path, str(starting_subject_number + i))
#     if not os.path.exists(output_subject_path):
#         os.mkdir(output_subject_path)
    
#     filenames = os.listdir(subject_path)
#     for file in filenames:
#         image = cv2.imread(os.path.join(subject_path, file))
#         boxes = face_recognition.face_locations(image, 1, 'hog')
#         for top, right, bottom, left in boxes:
#             face = image[top : bottom, left : right]
#             cv2.imwrite(os.path.join(output_subject_path, file), face)


input_paths = ['data/database collage/DB cam 1 version 3', 'data/database collage/DB cam 1 version 4', 'data/database collage/DB cam 1 version 5']
output_paths = ['data/database collage/detections/DB version 3', 'data/database collage/detections/DB version 4', 'data/database collage/detections/DB version 5']
for input_path, output_path in zip(input_paths, output_paths):
    utils.extract_faces_from_database(input_path, output_path, 0.5)