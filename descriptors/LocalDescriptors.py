import numpy as np, cv2
from skimage.feature import local_binary_pattern
from descriptors.weber import weber

class LocalPattern:
    def __init__(self):
        pass

    def preprocess(self, image, grid_x = 8, grid_y = 8):
        slices = []
        h, w = image.shape[:2]
        space_x, space_y = int(w / grid_x), int(h / grid_y)
        
        for y in range(grid_y):
            for x in range(grid_x):
                Y = y * space_y
                X = x * space_x
                slices.append(image[Y : Y + space_y, X : X + space_x])
        
        return slices
            
    def describe(self, image, eps = 1e-7):
        raise NotImplementedError()
       

class LocalBinaryPattern(LocalPattern):
    def __init__(self, num_points, radius):
        self.num_points = num_points
        self.radius = radius

    def describe(self, image, eps = 1e-7):
        slices = self.preprocess(image, 2, 2)
        result = []
        for i in range(len(slices)):
                
            lbp = local_binary_pattern(slices[i], self.num_points, self.radius, method = 'uniform')
            (hist, _) = np.histogram(lbp.ravel(), np.arange(0, self.num_points + 3), (0, self.num_points + 2))
            hist = hist.astype('float')
            hist /= (hist.max() + eps)
            
            result.extend(hist)
        
        return np.array(result)

class WeberPattern(LocalPattern):
    def __init__(self):
        pass
    
    def describe(self, image):
        slices = self.preprocess(image, 7, 7)
        result = []
        
        for  i in range(len(slices)):
            w = weber(slices[i])
            result.extend(w)
        
        return np.array(result)
