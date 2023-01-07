import numpy as np, cv2
from scipy.signal import convolve2d
import numba

@numba.njit()
def weber_histogram(m_bins, t_bins, differential_excitation, orientation, M, T, S, weights :np.ndarray = np.array([0.2688, 0.0852, 0.0955, 0.1000, 0.1018, 0.3487], dtype = np.float32)):
        histogram2d = np.zeros((M*S, T))
        for i in range(M):
            for k in range(S):
                for j in range(T):
                    condition1 = np.bitwise_and(differential_excitation >= m_bins[i], differential_excitation < m_bins[i + 1])
                    condition2 = np.bitwise_and(orientation >= t_bins[j], orientation < t_bins[j + 1])
                    s = np.floor((differential_excitation - m_bins[i]) / ((m_bins[i+1] - m_bins[i])/S) + 1/2)
                    condition3 = s == k
                    condition = np.bitwise_and(np.bitwise_and(condition1, condition2), condition3)
                    histogram2d[i*S + k, j] = np.sum(condition, dtype = np.float32) * weights[i]
        return histogram2d
    
def weber(grayscale_image):
    T = 8 #number of dominant orientations
    M = 6 #number of bins in the differential_excitation
    S = 4 #number of sub histograms
    grayscale_image = grayscale_image.astype(np.float64)
    grayscale_image[grayscale_image==0] = np.finfo(float).eps
    h, w = grayscale_image.shape[:2]
    neighbours_filter = np.array([
        [1,1,1],
        [1,-8,1],
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
    t_bins = np.array([2 * np.pi * t / T for t in range(T+1)])
    
    convolved = convolve2d(grayscale_image, neighbours_filter, mode='same')
    differential_excitation = convolved/grayscale_image
    differential_excitation = np.arctan(differential_excitation)
    m_bins = np.array([(m / M - 1/2) * np.pi for m in range(M + 1)])
    
    histogram2d = weber_histogram(m_bins, t_bins, differential_excitation, orientation, M, T, S)
    histogram2d = histogram2d / (histogram2d.max() + 1e-7) 
    return histogram2d.ravel()