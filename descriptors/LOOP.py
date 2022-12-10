import numba, numpy as np, cv2, time

mask = np.zeros((3, 3, 8))
mask[:, :, 0] = [
    [-3, -3, 5],
    [-3, 0, 5],
    [-3, -3, 5]
]

mask[:, :, 1] = [
    [-3, 5, 5],
    [-3, 0, 5],
    [-3, -3, -3]
]

mask[:, :, 2] = [
    [5, 5, 5],
    [-3, 0, -3],
    [-3, -3, -3]
]

mask[:, :, 3] = [
    [5, 5, -3],
    [5, 0, -3],
    [-3, -3, -3]
]

mask[:, :, 4] = [
    [5, -3, -3],
    [5, 0, -3],
    [5, -3, -3]
]

mask[:, :, 5] = [
    [-3, -3, -3],
    [5, 0, -3],
    [5, 5, -3]
]

mask[:, :, 6] = [
    [-3, -3, -3],
    [-3, 0, -3],
    [5, 5, 5]
]

mask[:, :, 7] = [
    [-3, -3, -3],
    [-3, 0, 5],
    [-3, 5, 5]
]

@numba.njit()
def loop(image):
    h, w = image.shape
    x = np.zeros(8)
    y = np.zeros(8)
    loop_image = np.zeros((h, w))
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            t = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    if k != 0 or l != 0:
                        if (image[i + k, j + l] - image[i, j]) < 0:
                            x[t] = 0
                        else:
                            x[t] = 1
                        y[t] = image[i + k, j + l] * mask[1 + k, 1 + l, t]
                        t = t + 1
                        
            p, q = np.sort(y), np.argsort(y)

            for t in range(8):
                loop_image[i, j] = loop_image[i, j] + ((2 ** (q[t])) * x[t])
    
    return loop_image.astype(np.uint8)
            
                
            
            
img = cv2.imread('data/images/Untitled.png', cv2.IMREAD_GRAYSCALE) / 256

t1 = time.time()
ldp_image = loop(img)
t2 = time.time()
print(t2 - t1)

cv2.imshow('ldp_image', ldp_image)
cv2.waitKey(0)


