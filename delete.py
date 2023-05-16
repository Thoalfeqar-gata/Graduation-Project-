import cv2, numpy as np, os
path = 'data/results2'
for filename in os.listdir(path):
    if 'Confusion' in filename:
        image = cv2.imread(os.path.join(path, filename))[190:1470, 730:2015]
        cv2.imwrite(os.path.join('data', 'results2', 'cropped', filename), image)
    elif 'ROC' in filename:
        image = cv2.imread(os.path.join(path, filename))[108:1005, 187:1732]
        cv2.imwrite(os.path.join('data', 'results2', 'cropped', filename), image)
        