import cv2, albumentations as A, os, numpy as np, numba, math, time
from tqdm import tqdm

transform = A.Compose([
    A.HorizontalFlip(p = 0.8),
    A.CLAHE(p = 0.8),
    A.RandomBrightnessContrast(brightness_limit = 0.2, contrast_limit = 0.1, p = 0.8),
    # A.RandomGamma(gamma_limit = (50, 80), p = 0.5),
    A.GaussianBlur(blur_limit = (3, 3), p = 0.5),
    # A.Emboss(alpha = (0.1, 0.3), strength = (0.1, 0.4), p = 0.5),
    # A.FancyPCA(p = 0.5),
    A.GaussNoise(p = 0.5),
    A.ImageCompression(quality_lower = 99, compression_type = A.augmentations.transforms.ImageCompression.ImageCompressionType.JPEG, p = 0.5),
    # A.ISONoise(intensity = (0.1, 0.15), p = 0.5),
    # A.PixelDropout(p = 0.5),
    # A.RandomShadow(shadow_dimension = 3, p = 0.5),
    # A.RandomToneCurve(p = 0.5),
    # A.Sharpen(p = 0.5),
    # A.Posterize(num_bits = 7, p = 0.5),
    # A.InvertImg(),
    # A.Affine(scale = (0.9, 1.1), translate_percent = (0, 0.05), rotate = (0, 5)),
    # A.CoarseDropout(max_holes = 4),
    A.MedianBlur(blur_limit = 3),
    A.Resize(144, 144, cv2.INTER_CUBIC, p = 0.5),
    # A.MultiplicativeNoise()
])

path = os.path.join('data', 'database collage', 'detections', 'DB unified', 'all faces')
output_path = os.path.join('data', 'database collage', 'detections', 'DB unified', 'all faces with augmentation')
increase_amount = 25
maximum_images_per_subject = 1000
subjects = os.listdir(path)

for index in tqdm(range(len(subjects))):
    subject = subjects[index]
    images = os.listdir(os.path.join(path, subject))
    if not os.path.exists(os.path.join(output_path, subject)):
        os.mkdir(os.path.join(output_path, subject))
  
    for i in range(len(images)):
        image_output_path = os.path.join(output_path, subject, images[i])
        image = cv2.imread(os.path.join(path, subject, images[i]))
        cv2.imwrite(image_output_path, image)
        
    if len(images) >= maximum_images_per_subject:
        continue
    
    generated_images = []
    for image in images:
        copies = []
        img_path = os.path.join(path, subject, image)
        img = cv2.imread(img_path)
        while len(copies) < increase_amount:
            new_img = transform(image = img)['image']
            
            if np.array_equal(new_img, img):
                continue
            for new_img in copies:
                if np.array_equal(new_img, img):
                    continue
            copies.append(new_img)
            
        generated_images.extend(copies)
        if (len(generated_images) + len(images)) >= maximum_images_per_subject:
            break
            
    for i in range(len(generated_images)):
        image_output_path = os.path.join(output_path, subject, f'AUG_{time.time()}.jpg')
        cv2.imwrite(image_output_path, generated_images[i])
    

