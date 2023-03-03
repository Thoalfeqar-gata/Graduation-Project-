import cv2, albumentations as A, os, numpy as np, numba, math
from tqdm import tqdm

transform = A.Compose([
    A.HorizontalFlip(p = 0.7),
    A.RandomBrightnessContrast(brightness_limit = 0.3, contrast_limit = 0.1, p = 0.8),
    A.RandomGamma(gamma_limit = (50, 80), p = 0.5),
    A.GaussianBlur(blur_limit = (3, 3), p = 0.5),
    A.Emboss(alpha = (0.1, 0.3), strength = (0.1, 0.4), p = 0.5),
    A.CLAHE(p = 0.8),
    A.Equalize(p = 0.4),
    # A.FancyPCA(p = 0.5),
    A.GaussNoise(p = 0.5),
    A.ImageCompression(quality_lower = 99, compression_type = A.augmentations.transforms.ImageCompression.ImageCompressionType.JPEG, p = 0.5),
    A.ISONoise(intensity = (0.1, 0.15), p = 0.5),
    A.PixelDropout(p = 0.5),
    A.RandomShadow(shadow_dimension = 3, p = 0.5),
    # A.RandomToneCurve(p = 0.5),
    A.Sharpen(p = 0.5),
    A.Posterize(num_bits = 7, p = 0.5),
    # A.InvertImg(),
    A.Affine(scale = (0.9, 1.1), translate_percent = (0, 0.05), rotate = (0, 5)),
    A.CoarseDropout(max_holes = 4),
    A.MedianBlur(blur_limit = 3),
    A.Resize(144, 144, cv2.INTER_CUBIC, p = 0.5),
    A.MultiplicativeNoise()
])

path = os.path.join('data', 'lfw', 'lfw_funneled')
output_path = os.path.join('data', 'lfw', 'lfw augmented')
increase_amount = 100
maximum_images_per_subject = 1000
subjects = os.listdir(path)

for index in tqdm(range(len(subjects))):
    subject = subjects[index]
    images = os.listdir(os.path.join(path, subject))
    if not os.path.exists(os.path.join(output_path, subject)):
        os.mkdir(os.path.join(output_path, subject))

    images_to_generate = len(images) * increase_amount
    images_to_generate = maximum_images_per_subject if images_to_generate >= maximum_images_per_subject else images_to_generate
    generated_images = []
    names = []
    while len(generated_images) <= images_to_generate:
        for image in images:
            img_path = os.path.join(path, subject, image)
            names.append(image)
            img = cv2.imread(img_path)
            new_img = transform(image = img)['image']
            
            if np.all(new_img == img):
                continue
            for old_img in generated_images:
                if np.all(old_img == new_img):
                    continue
            
            generated_images.append(new_img)
            
    for i in range(len(generated_images)):
        image_output_path = os.path.join(output_path, subject, f'AUG_{i}_{names[i]}')
        cv2.imwrite(image_output_path, generated_images[i])
    
    for i in range(len(images)):
        image_output_path = os.path.join(output_path, subject, names[i])
        image = cv2.imread(os.path.join(path, subject, names[i]))
        cv2.imwrite(image_output_path, image)