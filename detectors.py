import cv2, face_recognition, mediapipe as mp

image = cv2.imread('data/images/people.jpg')
opencv_detector = cv2.CascadeClassifier('data/cascades/haarcascade_frontalface_default.xml')

dets_opencv = opencv_detector.detectMultiScale(image, 1.1, 4)

opencv_image = image.copy()
for x, y, w, h in dets_opencv:
    cv2.rectangle(opencv_image, (x, y), (x + w, y + h), (0, 0, 255), 2)
    
dlib_image_hog = image.copy()
dets_dlib_hog = face_recognition.face_locations(dlib_image_hog, 1, 'hog')
for top, right, bottom, left in dets_dlib_hog:
    cv2.rectangle(dlib_image_hog, (left, top), (right, bottom), (0, 0, 255), 2)
    
dlib_image_cnn = image.copy()
dets_dlib_cnn = face_recognition.face_locations(dlib_image_hog, 1, 'cnn')
for top, right, bottom, left in dets_dlib_cnn:
    cv2.rectangle(dlib_image_cnn, (left, top), (right, bottom), (0, 0, 255), 2)

cv2.imwrite('opencv.jpg', opencv_image)
cv2.imwrite('dlib_hog.jpg', dlib_image_hog)
cv2.imwrite('dlib_cnn.jpg', dlib_image_cnn)