import cv2
import numpy as np
from ex1_utils import show_img
from ex2_utils import generate_responses_1, get_patch, show_mean_shift_progression


def mean_shift(img, patch_center, kernel_size, conv_threshold = 0.01):
    assert kernel_size[0] % 2 != 0 and kernel_size[1] % 2 != 0, f'Kernel size should be odd number!'
    xi, yi = create_kernels(kernel_size)
    x_step = 1
    y_step = 1
    current_patch_center = patch_center
    
    while abs(x_step) > conv_threshold and abs(y_step) > conv_threshold:
        patch, _ = get_patch(img, current_patch_center, kernel_size)
        
        x_step = np.divide(np.sum(np.multiply(xi, patch)), np.sum(patch))
        y_step = np.divide(np.sum(np.multiply(yi, patch)), np.sum(patch))
        
        new_patch_center = current_patch_center[0] + x_step, current_patch_center[1] + y_step
        current_patch_center = new_patch_center
        
        show_mean_shift_progression(img, patch, patch_center, new_patch_center)
        
    print(current_patch_center)
    print(patch.shape)
    show_img(patch, True)
    

def create_kernels(kernel_size):
    xi = np.zeros(kernel_size)
    yi = np.zeros(kernel_size)
    
    kernel_range = (kernel_size[0]-1)/2
    weights = np.arange(-kernel_range, kernel_range + 1)
    for i in range(xi.shape[0]):
        xi[:, i] = weights[i]
        yi[i, :] = weights[i]
    
    return xi, yi


if __name__ == "__main__":
    response = generate_responses_1()
    mean_shift(response, (40,60), (7,7))
    #show_img(response, normalized=True)