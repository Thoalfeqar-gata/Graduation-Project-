import numpy as np, cv2
from scipy.signal import convolve2d


def weber(grayscale_image):
    T = 8 #number of dominant orientations
    M = 6 #number of bins in the differential_excitation
    grayscale_image = grayscale_image.astype(np.float64)
    grayscale_image[grayscale_image==0] = np.finfo(float).eps
    neighbours_filter = np.array([
        [1,1,1],
        [1,0,1],
        [1,1,1]
    ])
    vertical_gradient = np.array([
        [0, -1, 0],
        [0, 0, 0],
        [0, 1, 0]
    ])
    horizontal_gradient = np.array([
        [0, 0, 0],
        [1, 0, -1],
        [0, 0, 0]
    ])
    convolved_vertical = convolve2d(grayscale_image, vertical_gradient, mode = 'same')
    convolved_horizontal = convolve2d(grayscale_image, horizontal_gradient, mode = 'same')
    convolved_vertical[convolved_vertical == 0] = np.finfo(float).eps
    phi = np.arctan2(convolved_horizontal, convolved_vertical) + np.pi
    #quantization
    t = np.mod(np.floor((phi / (2 * np.pi / T)) + 1/2), T)
    orientation = (2 * np.pi *  t)/T
    t_bins = [2 * np.pi * t / T for t in range(T + 1)]
    
    convolved = convolve2d(grayscale_image,neighbours_filter, mode='same')
    differential_excitation = convolved-8*grayscale_image
    differential_excitation = differential_excitation/grayscale_image
    differential_excitation = np.arctan(differential_excitation) + np.pi/2
    m_bins = [(m / M - 1/2) * np.pi for m in range(M + 1)]
    
    hist, _, _ = np.histogram2d(orientation.ravel(), differential_excitation.ravel(), bins = (t_bins, m_bins))
    hist = hist / (hist.max() + 1e-7)
    return hist.ravel()