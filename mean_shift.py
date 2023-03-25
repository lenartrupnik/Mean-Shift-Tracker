import cv2, math
import numpy as np
from ex1_utils import show_img
from ex2_utils import *
GROUND_TRUTH = [(70,50), (50,70)]

def mean_shift(img, patch_center= (50,50), kernel_size=(10,10), conv_threshold = 0.1, analyze=False):
    """
    Parameters: 
    img - input image with some 
    ----------------------------
    Return:
    iterations, convergence location
    """
    assert kernel_size[0] % 2 != 0 and kernel_size[1] % 2 != 0, f'Kernel size should be odd number!'
    xi, yi = create_kernels(kernel_size)
    x_step = 1
    y_step = 1
    current_patch_center = patch_center
    i = 0
    
    while abs(x_step) > conv_threshold or abs(y_step) > conv_threshold:
        patch, _ = get_patch(img, current_patch_center, kernel_size)
        
        x_step = np.divide(np.sum(np.multiply(xi, patch)), np.sum(patch))
        y_step = np.divide(np.sum(np.multiply(yi, patch)), np.sum(patch))
        
        new_patch_center = current_patch_center[0] + x_step, current_patch_center[1] + y_step
        current_patch_center = new_patch_center
        i +=1
        
        #show_mean_shift_progression(img * 255, patch, patch_center, new_patch_center)
        
        if i > 500:
            break
    #show_img(patch , True)
    if analyze:
        return i, current_patch_center
    

def create_kernels(kernel_size):
    xi = np.zeros((kernel_size[1], kernel_size[0]))
    yi = np.zeros((kernel_size[1], kernel_size[0]))
    
    width = (kernel_size[1]-1)/2
    height = (kernel_size[0]-1)/2
    weights_width = np.arange(-width, width + 1)
    weights_height = np.arange(-height, height + 1)
    for i in range(xi.shape[1]):
        xi[:, i] = weights_height[i]
        
    for i in range(yi.shape[0]):
        yi[i, :] = weights_width[i]
    
    return xi, yi


def analyze_parameters(image):
    kernel_size = [(5,5), (9,9), (31, 31)]
    starting_point = [(0,0), (20,70), (80, 20)]
    term_criteria = [0.1, 0.01, 0.5]

    for kernel in kernel_size:
        start = (35,40)
        iterations, conv_loc = mean_shift(image, start, kernel, 0.05, True)
        conv_loc = (np.round(conv_loc[0], 2), np.round(conv_loc[1], 2))
        error = calc_error(conv_loc)
        write_results(kernel, start, 0.05, iterations, conv_loc, error)
        print(f'>>>Finished convergence in {iterations} iterations!')
        
    for point in starting_point:
        kernel = (7,7)
        iterations, conv_loc = mean_shift(image, point, kernel, term_criteria[1], True)
        conv_loc = (np.round(conv_loc[0], 2), np.round(conv_loc[1], 2))
        error = calc_error(conv_loc)
        write_results(kernel, point, 0.05, iterations, conv_loc, error)
        print(f'>>>Finished convergence in {iterations} iterations!')
        
    for thresh in term_criteria:
        kernel = (7,7)
        start = (40,40)
        iterations, conv_loc = mean_shift(image, start, kernel, thresh, True)
        conv_loc = (np.round(conv_loc[0], 2), np.round(conv_loc[1], 2))
        error = calc_error(conv_loc)
        write_results(kernel, start, thresh, iterations, conv_loc, error)
        print(f'>>>Finished convergence in {iterations} iterations!')


def write_results(kernel, start, threshold, iteration, convergence_loc, error):
    with open("results.txt", '+a') as f:
        f.write(f'\n{kernel}, {start}, {threshold}, {iteration}, {convergence_loc}, {error}')
        f.close
        
        
def calc_error(location):
    distances = []
    for truth in GROUND_TRUTH:
        distances.append(np.round(math.dist(location, truth), 2))
    return min(distances)


def plot_custom():
    response = generate_responses_2()
    start = (35, 41)
    _, end_0 = mean_shift(response, start, (5,5), 0.05, True)
    _, end_1 = mean_shift(response, start, (39,39), 0.5, True)
    
    img = cv2.cvtColor(response*255, cv2.COLOR_GRAY2RGB)
    img = cv2.circle(img, start, 1, (0,255,0), 2)
    img = cv2.circle(img, (int(end_0[0]), int(end_0[1])), 1, (0,0,255), 2)
    img = cv2.circle(img, (int(end_1[0]), int(end_1[1])), 1, (0,150,150), 2)
    img = cv2.circle(img, (70, 10), 1, (0,255,0), 2)
    img = cv2.circle(img, (70, 20), 1, (0,0,255), 2)
    img = cv2.circle(img, (70, 30), 1, (0,150,150), 2)
    show_img(img, True)
    cv2.imwrite("New setup.png", img)

if __name__ == "__main__":
    response = generate_responses_1()
    #show_img(response*255, True)
    mean_shift(response, (50,50), (51,51), 0.5)
    #analyze_parameters(response)
    #plot_custom()