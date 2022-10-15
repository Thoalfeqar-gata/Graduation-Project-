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

@numba.njit
def ldp(image):
    h, w = image.shape
    z = np.zeros(8)
    ldp_image = np.zeros((h, w))
    
    for i in range(1, h - 1):
        for j in range(1, w - 1):
            t = 0
            for k in range(-1, 2):
                for l in range(-1, 2):
                    if k != 0 or l != 0:
                        z[t] = image[i + k, j + l] * mask[1 + k, 1 + l, t]
                        t = t + 1
            p, q = np.sort(z), np.argsort(z)
            g = 4
            
            for t in range(g):
                z[q[t]] = 0
            for t in range(g, 8):
                z[q[t]] = 1 
            
            for t in range(8):
                ldp_image[i, j] = ldp_image[i, j] + ((2 ** t) * z[t])
    
    return ldp_image.astype(np.uint8)
            
                
            
            
img = cv2.imread('data/images/eye.jpg', cv2.IMREAD_GRAYSCALE) / 256

t1 = time.time()
ldp_image = ldp(img)
t2 = time.time()
print(t2 - t1)

cv2.imshow('ldp_image', ldp_image)
cv2.waitKey(0)


