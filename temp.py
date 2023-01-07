import cv2
import numpy as np

# Read the input image
image = cv2.imread('data/images/image.jpg')

# Convert the image to grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Canny edge detection
edges = cv2.Canny(gray, 100, 200)

# Display the resulting image
cv2.imshow('Edges', edges)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('data/images/canny_image.jpg', edges)