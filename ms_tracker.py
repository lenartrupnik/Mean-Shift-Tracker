import numpy as np
from ex2_utils import *
from mean_shift import create_kernels

class MeanShiftTracker(Tracker):
    def __init__(self, params):
        super().__init__(params)

        
    def initialize(self, image, region, background_norm = False):
        if len(region) == 8:
            x_ = np.array(region[::2])
            y_ = np.array(region[1::2])
            region = [np.min(x_), np.min(y_), np.max(x_) - np.min(x_) + 1, np.max(y_) - np.min(y_) + 1]
        
        region[2] = math.floor(region[2])
        if region[2] % 2 == 0:
            region[2] -= 1
        
        region[3] = math.floor(region[3])
        if region[3] % 2 == 0:
            region[3] -= 1
            
        self.window = max(region[2], region[3]) * self.parameters.enlarge_factor
        left = max(region[0], 0)
        top = max(region[1], 0)
        
        right = min(region[0] + region[2], image.shape[1] - 1)
        bottom = min(region[1] + region[3], image.shape[0]  -1)
        self.size = (int(region[2]), int(region[3]))
        patch = image[int(top):int(bottom), int(left):int(right)]
        self.position = (region[0] + region[2] / 2, region[1] + region[3] / 2)
        self.norm_back = background_norm
        #show_img(patch)
        self.c = norm_hist(image, self.position, self.parameters.bins, self.size) if background_norm else None
        self.template = patch
        his_ = extract_histogram(patch, self.parameters.bins)
        self.q =  his_ * self.c if background_norm else his_
        self.kernel = create_epanechnik_kernel(self.size[0], self.size[1], self.parameters.sigma)

        
    def track(self, image, update_q = True):
        xi, yi = create_kernels(self.size)
        left = max(round(self.position[0] - float(self.window) / 2), 0)
        top = max(round(self.position[1] - float(self.window) / 2), 0)

        right = min(round(self.position[0] + float(self.window) / 2), image.shape[1] - 1)
        bottom = min(round(self.position[1] + float(self.window) / 2), image.shape[0] - 1)
        
        if right - left < self.template.shape[1] or bottom - top < self.template.shape[0]:
            return [self.position[0] + self.size[0] / 2, self.position[1] + self.size[1] / 2, self.size[0], self.size[1]]

        #Include mean shift
        converged = False
        N = 0
        while not converged and N < 20:
            patch, _ = get_patch(image, self.position, self.size)
            #show_img(patch)
            his_ = extract_histogram(patch, self.parameters.bins, weights = self.kernel)
            p = self.c * his_ if self.norm_back else his_
            v = np.sqrt(np.divide(self.q, p + self.parameters.epsilon))
            w = backproject_histogram(patch, v, self.parameters.bins)
            #show_img(w)
            x_step = np.divide(np.sum(np.multiply(xi, w)), np.sum(w))
            y_step = np.divide(np.sum(np.multiply(yi, w)), np.sum(w))
            
            if abs(x_step) < self.parameters.threshold and abs(y_step) < self.parameters.threshold:
                converged = True
            
            if math.isnan(x_step) or math.isnan(y_step):
                break
            
            self.position = self.position[0] + x_step, self.position[1] + y_step
            N += 1
        
        if update_q:
            self.q = 0.95 * +   self.q + 0.05 * p
    
        return self.position[0] - self.size[0] / 2, self.position[1] - self.size[1] / 2, self.size[0], self.size[1]