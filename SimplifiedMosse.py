import numpy as np
from ex2_utils import *
from ex3_utils import *
from numpy.fft import fft2, ifft2

class CorelationParams():
    def __init__(self, enlarge_factor=1, sigma = 4, lmbd = 1, alfa = 0.1):
        self.enlarge_factor = enlarge_factor
        self.sigma = sigma
        self.lmbd = lmbd
        self.alpha = alfa

class CorrelationTracker(Tracker):
    def __init__(self, params):
        self.params = CorelationParams()
        super().__init__(params)
        
    def initialize(self, img, region: list):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
        
        # Locate patch and location of searched area
        self.window = max(region[2], region[3]) * self.params.enlarge_factor
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        
        # Define search region around image with some enlarged factor
        self.size= (region[2], region[3])
        
        # Construct gaussian function
        self.G = create_gauss_peak((int(self.window), int(self.window)), self.params.sigma)
        self.shape = self.G.shape
        self.G = fft2(self.G)
        
        # Prepare feature patch
        self.cosine_window = create_cosine_window(self.shape)
        patch, _ = get_patch(img, self.position, self.shape)
        patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        patch = np.multiply(patch_gray, self.cosine_window)
        self.F = fft2(patch)
        self.F_conj = np.conjugate(self.F)
        
        # Construct filter

        filter_fft = np.divide(
            np.multiply(self.G, self.F_conj),
            np.add(self.params.lmbd, np.multiply(self.F, self.F_conj))
        )
        
        self.H_conj = np.conjugate(filter_fft)
    
    def track(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        patch, _ = get_patch(img, self.position, self.shape)
        patch = fft2(np.multiply(patch, self.cosine_window))
        
        # Localization
        R = ifft2(np.multiply(patch, self.H_conj))
        y_max, x_max = np.unravel_index(R.argmax(), R.shape)
        
        # Compensate for inverted Gaussian
        if x_max >= patch.shape[0] / 2:
            x_max = x_max - patch.shape[0]
        if y_max >= patch.shape[1] / 2:
            y_max = y_max - patch.shape[1]
            
        # Update new location
        self.position = (self.position[0] + x_max, self.position[1] + y_max)
        
        # Update filter
        patch_new, _ = get_patch(img, self.position, self.shape)
        patch_new = fft2(np.multiply(patch_new, self.cosine_window))
        patch_new_conj = np.conjugate(patch_new)
        new_filter = np.divide(
            np.multiply(self.G, patch_new_conj),
            np.add(self.params.lmbd, np.multiply(patch_new,patch_new_conj))
        )
        
        self.H_conj = (1 - self.params.alpha) * self.H_conj + self.params.alpha * new_filter
        
        return self.position[0] - (self.size[0] / 2), self.position[1] - self.size[1] / 2, self.size[0], self.size[1]