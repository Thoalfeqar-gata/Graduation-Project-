import cv2, albumentations as A, os, numpy as np, numba, math

transform = A.Compose([
    A.HorizontalFlip(p = 0.5),
    A.RandomBrightnessContrast(p = 0.5),
    A.CLAHE(p = 0.5),
    A.RandomGamma(p = 0.5),
    A.ShiftScaleRotate(p = 0.5, rotate_limit = 15),
    A.GaussianBlur(blur_limit = (3, 3), p = 0.5)
])

path = os.path.join('data', 'database collage', 'detections', 'all faces')
output_path = os.path.join('data', 'database collage', 'detections', 'all faces with augmentation')
for i in range(36):
    os.mkdir(os.path.join(output_path, str(i)))
    
for dirname, dirnames, filenames in os.walk(path):
    images_per_subject = 400
    augmented_and_original = {}
    original_images = {}

    if len(filenames) <= 0:
        continue

    for filename in filenames:
        image_path = os.path.join(dirname, filename)
        image = cv2.imread(image_path)
        original_images[filename] = image
    
    augmented_and_original = original_images.copy()
    copies_from_each_image = math.ceil(400 / len(original_images))
    
    for key in original_images.keys():
        for i in range(copies_from_each_image):
            augmented = transform(image = original_images[key])['image']
            augmented_and_original[f'AUG_{i}_' + key] = augmented
    
    images_folder = os.path.split(dirname)[1]
    for key in augmented_and_original.keys():
        image = augmented_and_original[key]
        p = os.path.join(output_path, images_folder, key)
        cv2.imwrite(p, image)   
    
            
    
    
    
            
    
    
            

