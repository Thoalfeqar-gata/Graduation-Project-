import cv2, numpy as np, face_recognition

image = cv2.imread('data/images/people.jpg')
hog_image = cv2.imread('data/images/hog.jpg')
gist_image = cv2.imread('data/images/gist.jpg')
boxes = face_recognition.face_locations(image, model = 'cnn')

i = 0
for top, right, bottom, left in boxes:
    face_hog = hog_image[top:bottom, left:right]
    face_gist = gist_image[top:bottom, left:right]
    cv2.imshow('hog', face_hog)
    cv2.imshow('gist', face_gist)
    cv2.waitKey(0)