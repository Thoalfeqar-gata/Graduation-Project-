import cv2, albumentations as A, os, numpy as np, numba, math
from tqdm import tqdm

transform = A.Compose([
    A.HorizontalFlip(p = 0.5),
    A.RandomBrightnessContrast(p = 0.5),
    A.RandomGamma(p = 0.5),
    A.GaussianBlur(blur_limit = (3, 3), p = 0.5),
    A.Emboss(p = 0.5),
    A.CLAHE(),
    A.ChannelShuffle(),
    A.ColorJitter(0.1, 0.1, 0.1, 0.1)
])

path = os.path.join('data', 'database collage', 'detections', 'DB unified', 'all faces')
output_path = os.path.join('data', 'database collage', 'detections', 'DB unified', 'all faces with augmentation')
images_per_subject = 200
subjects = os.listdir(path)

for index in tqdm(range(len(subjects))):
    subject = subjects[index]
    images = os.listdir(os.path.join(path, subject))
    if not os.path.exists(os.path.join(output_path, subject)):
        os.mkdir(os.path.join(output_path, subject))
        
    if len(images) < images_per_subject:
        images_to_generate = images_per_subject - len(images)
        
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