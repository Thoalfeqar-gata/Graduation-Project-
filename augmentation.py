import cv2, albumentations as A, os, numpy as np, numba, math
from tqdm import tqdm

transform = A.Compose([
    A.HorizontalFlip(p = 0.5),
    A.RandomBrightnessContrast(p = 0.5),
    A.CLAHE(p = 0.5),
    A.RandomGamma(p = 0.5),
    A.ShiftScaleRotate(p = 0.5, rotate_limit = 15),
    A.GaussianBlur(blur_limit = (3, 3), p = 0.5),
    A.GaussNoise((5, 25), p = 0.5),
    A.Sharpen(p = 0.5),
    A.Emboss(p = 0.5)
])

path = os.path.join('data', 'lfw_funneled')



images_per_subject = 50
subjects = os.listdir(path)

for index in tqdm(range(len(subjects))):
    subject = subjects[index]
    images = os.listdir(os.path.join(path, subject))
    
    if len(images) < images_per_subject:
        images_to_generate = images_per_subject - len(images)
        
        generated_images = []
        
        while len(generated_images) <= images_to_generate:
            for image in images:
                img_path = os.path.join(path, subject, image)
                img = cv2.imread(img_path)[13:250-13, 13:250-13]
                
                new_img = transform(image = img)['image']
                
                if np.all(new_img == img):
                    continue
                for old_img in generated_images:
                    if np.all(old_img == new_img):
                        continue
                
                generated_images.append(new_img)
    
        for  i in range(len(generated_images)):
            output_path = os.path.join('data', 'lfw augmented', subject, f'AUG_{i}.jpg')
            cv2.imwrite(output_path, generated_images[i])
            
        
          
        