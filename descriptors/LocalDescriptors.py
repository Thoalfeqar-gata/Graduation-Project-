import numpy as np, cv2
from tqdm import tqdm
from skimage.feature import local_binary_pattern
from descriptors.weber2 import weber

class LocalPattern:
    def __init__(self, grid_shape = (6, 6)):
        self.grid_shape = grid_shape

    def preprocess(self, image):
        slices = []
        grid_x, grid_y = self.grid_shape
        h, w = image.shape[:2]
        space_x, space_y = int(w / grid_x), int(h / grid_y)
        
        for y in range(grid_y):
            for x in range(grid_x):
                Y = y * space_y
                X = x * space_x
                slices.append(image[Y : Y + space_y, X : X + space_x])
        
        return np.array(slices)
            
    def compute(self, image, eps = 1e-7):
        raise NotImplementedError()
       

class LocalBinaryPattern(LocalPattern):
    def __init__(self, num_points, radius, grid_shape):
        self.num_points = num_points
        self.radius = radius
        super().__init__(grid_shape)

    def compute(self, images, eps = 1e-7):
        features = []
        print('Processing lbp features...')
        for i in tqdm(range(len(images))):
            image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            slices = self.preprocess(image)
            result = []
            for i in range(len(slices)):
                    
                lbp = local_binary_pattern(slices[i], self.num_points, self.radius, method = 'uniform')
                (hist, _) = np.histogram(lbp.ravel(), np.arange(0, self.num_points + 3), (0, self.num_points + 2))
                hist = hist.astype('float')
                hist /= (hist.max() + eps)
                
                result.extend(hist)
            features.append(result)
        return features

class WeberPattern(LocalPattern):
    def __init__(self, grid_shape):
        super().__init__(grid_shape)
    
    def compute(self, images):
        features = []
        print('Processing weber features...')
        for i in tqdm(range(len(images))):
            image = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            if self.grid_shape == (1, 1):
                slices = [image]
            else:
                slices = self.preprocess(image)
            result = []
        
            for  i in range(len(slices)):
                w = weber(slices[i])
                result.extend(w)
                
            features.append(result)
            
        return features
