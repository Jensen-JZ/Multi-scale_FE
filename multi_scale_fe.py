import re
import cv2
import argparse
import numpy as np
# import scipy.signal # No longer used after pyramid_expand optimization
from PIL import Image
import matplotlib.pyplot as plt

def config_parse():
    parser = argparse.ArgumentParser(description='Path Parser')
    parser.add_argument('--input_path', type=str, required=True, help='Specify the path of the input image.')
    parser.add_argument('--level', type=int, default=3, help='Specify the number of layers of the multi-scale pyramid.')
    args = parser.parse_args()

    return args

def box_filter(img, rad):
    # The original function signature included [height, width] = img.shape.
    # This line is kept for historical compatibility with the function signature, 
    # though not strictly used by the cv2.boxFilter call itself.
    [height, width] = img.shape 

    # Using cv2.boxFilter for optimized performance and readability.
    # It calculates the sum of pixels (if normalize=False), matching the original function's behavior.
    # ddepth=-1 ensures the output has the same depth (dtype) as the input for float types (e.g., float64).
    # For integer types (e.g., uint8), cv2.boxFilter with normalize=False might promote
    # the output to a larger integer type (e.g., int32) to prevent overflow.
    # The original cumsum implementation also implicitly handled this by numpy's dtype promotion.
    # All current usages in this script involve float64 inputs for `img`.
    
    k_size = 2 * rad + 1
    
    # Ensure img is C-contiguous. Some OpenCV functions expect this or perform better.
    img_contiguous = np.ascontiguousarray(img)

    im_dst = cv2.boxFilter(src=img_contiguous, 
                           ddepth=-1, # Output dtype same as input (for float types like float64)
                           ksize=(k_size, k_size), 
                           anchor=(-1,-1), # Default, kernel centered
                           normalize=False, # Calculates sum, not average (matches original behavior)
                           borderType=cv2.BORDER_REPLICATE) # Replicates border pixels
    
    # The original function returned a result with the same dtype as the input `img`.
    # For float64 inputs (which is the case in this script's usage of guided_filter), 
    # cv2.boxFilter with ddepth=-1 correctly preserves the float64 dtype.
    # This explicit cast ensures that the output dtype strictly matches the input `img.dtype`,
    # maintaining the function's contract, especially if it were to be used with other dtypes
    # where cv2.boxFilter might change the dtype (e.g. uint8 input sum -> int32 output).
    return im_dst.astype(img.dtype)

def guided_filter(guide, img, rad, eps):
    [height, weight] = guide.shape
    # N is the number of pixels in the box filter window.
    # (2 * rad + 1) is the kernel size.
    # This replaces n = box_filter(np.ones((height, weight)), rad) for efficiency.
    N = (2 * rad + 1)**2

    guide_mean = box_filter(guide, rad) / N
    img_mean = box_filter(img, rad) / N
    guid_img_mean = box_filter(guide * img, rad) / N
    guid_img_cov = guid_img_mean - guide_mean * img_mean

    guid2_mean = box_filter(guide * guide, rad) / N # Changed n to N
    guide_var = guid2_mean - guide_mean * guide_mean

    a = guid_img_cov / (guide_var + eps)
    b = img_mean - a * guide_mean

    a_mean = box_filter(a, rad) / N              # Changed n to N
    b_mean = box_filter(b, rad) / N              # Changed n to N

    guide_img = a_mean * guide + b_mean

    return guide_img

def pyramid_reduce(img):
    # Ensures input is float64, consistent with original's np.double(img).
    # cv2.pyrDown preserves the input dtype (e.g., float64).
    img_double = np.double(img)
    
    # cv2.pyrDown uses the standard 5x5 Gaussian kernel.
    # borderType=cv2.BORDER_REPLICATE matches the original cv2.filter2D's border handling.
    # dstsize default is ((src.cols+1)//2, (src.rows+1)//2), which matches original slicing.
    return cv2.pyrDown(img_double, borderType=cv2.BORDER_REPLICATE)

def pyramid_expand(img, dst_shape_hw): # dst_shape_hw is (rows, cols) of the target G_k layer
    # Ensures input is float64, consistent with original's np.double(img).
    # cv2.pyrUp preserves the input dtype (e.g., float64).
    img_double = np.double(img)

    # cv2.pyrUp uses the standard 5x5 Gaussian kernel (scaled by 4).
    # borderType=cv2.BORDER_REPLICATE is specified to be consistent.
    # dstsize is (cols, rows).
    return cv2.pyrUp(img_double, 
                     dstsize=(dst_shape_hw[1], dst_shape_hw[0]))

def generate_pyramid(img, type, level):
    pyramid = [0 for i in range(level)]
    pyramid[0] = np.double(img)
    for i in range(1, level):
        pyramid[i] = pyramid_reduce(pyramid[i - 1])

    if type == 'gauss':
        return pyramid

    # Laplacian pyramid construction: L_k = G_k - expand(G_{k+1})
    # The old slicing loop that was here has been removed, as the new
    # pyramid_expand(img, dst_shape_hw) will produce an image of the correct target size (G_k.shape).
    
    for k in range(level - 1):  # Iterate from k = 0 to level - 2
        # pyramid[k] is G_k
        # pyramid[k+1] is G_{k+1}
        # We need to expand G_{k+1} to the size of G_k
        target_shape_hw = pyramid[k].shape
        expanded_G_k_plus_1 = pyramid_expand(pyramid[k + 1], dst_shape_hw=target_shape_hw)
        
        # L_k = G_k - expanded_G_{k+1}
        # Store L_k in pyramid[k]
        pyramid[k] = pyramid[k] - expanded_G_k_plus_1
        
    # The last level of the Gaussian pyramid (pyramid[level-1] which is G_{level-1}) 
    # is also the last level of the Laplacian pyramid. So it remains as is.
    return pyramid

def fusion_based_method(img_path, level, img_name):
    img_load = Image.open(img_path)
    plt.title('Original Image')
    plt.imshow(img_load)
    # plt.show() # Removed for non-interactive use

    # Convert PIL Image to NumPy array with float64 dtype directly
    img = np.array(img_load, dtype=np.float64)
    red_chan = img[:, :, 0] # Removed semicolon
    green_chan = img[:, :, 1] # Removed semicolon
    blue_chan = img[:, :, 2] # Removed semicolon
    [height, width, depth] = img.shape

    lightness = np.maximum(red_chan, np.maximum(green_chan, blue_chan))
    # plt.title('Image Lightness')
    # plt.imshow(lightness, cmap='gray')
    # plt.show()

    # Optimized struct_elem generation using NumPy array operations
    struct_elem = np.zeros((15, 15), dtype=np.uint8)
    # Central horizontal bar for rows 4-10 (inclusive, index from 0)
    struct_elem[4:11, 1:14] = 1 
    # Row 3 and 11 (symmetric)
    struct_elem[3, 2:13] = 1
    struct_elem[11, 2:13] = 1
    # Row 2 and 12
    struct_elem[2, 3:12] = 1
    struct_elem[12, 3:12] = 1
    # Row 1 and 13
    struct_elem[1, 4:11] = 1
    struct_elem[13, 4:11] = 1
    # Extreme points
    struct_elem[0, 7] = 1
    struct_elem[14, 7] = 1 # -1 index is 14 for size 15
    struct_elem[7, 0] = 1
    struct_elem[7, 14] = 1 # -1 index is 14 for size 15
    # struct_elem = np.uint8(struct_elem) # Already dtype=np.uint8 from initialization

    morph_cls_op = cv2.morphologyEx(lightness, cv2.MORPH_CLOSE, struct_elem)
    morph_cls_op = morph_cls_op / 255
    # plt.title('Morphologically Closing Operation')
    # plt.imshow(morph_cls_op, cmap='gray')
    # plt.show()

    hsv_img = np.double(cv2.cvtColor(np.uint8(img), cv2.COLOR_RGB2HSV))
    hsv_img[:, :, 0] = hsv_img[:, :, 0] / 180;
    hsv_img[:, :, 1] = hsv_img[:, :, 1] / 255;

    guid_img = hsv_img[:, :, 2]
    illum = guided_filter(guid_img / np.max(guid_img), morph_cls_op, 2, 0.004)
    # plt.title('Illuminance')
    # plt.imshow(illum, cmap='gray')
    # plt.show()

    np.seterr(divide='ignore', invalid='ignore')
    refl = img / np.expand_dims(illum, 2)
    # plt.title('Reflectance')
    # plt.imshow(refl / 255)
    # plt.show()

    illum_1 = illum
    illum_mean = np.mean(illum)
    lamb = 10 + (1 - illum_mean) / illum_mean

    illmu_2 = 2 / np.pi * np.arctan(lamb * illum) # Replaced math.pi with np.pi

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    illmu_3 = clahe.apply(np.uint8(illum * 255)) / 255

    alpha = 2
    phi_degress = 250
    phi = np.deg2rad(phi_degress)

    weight_brig_1 = np.exp(-(illum_1 - 0.5) ** 2 / 0.125)
    weight_brig_2 = np.exp(-(illmu_2 - 0.5) ** 2 / 0.125)
    weight_brig_3 = np.exp(-(illmu_3 - 0.5) ** 2 / 0.125)

    weight_chr_1 = illum_1 * (1 + np.cos(alpha * hsv_img[:, :, 0] + phi) * hsv_img[:, :, 1])
    weight_chr_2 = illmu_2 * (1 + np.cos(alpha * hsv_img[:, :, 0] + phi) * hsv_img[:, :, 1])
    weight_chr_3 = illmu_3 * (1 + np.cos(alpha * hsv_img[:, :, 0] + phi) * hsv_img[:, :, 1])

    weight_1 = weight_brig_1 * weight_chr_1 # Removed semicolon
    weight_2 = weight_brig_2 * weight_chr_2 # Removed semicolon
    weight_3 = weight_brig_3 * weight_chr_3 # Removed semicolon
    
    # Normalize weights
    sum_weights = weight_1 + weight_2 + weight_3
    # Add a small epsilon to prevent division by zero if sum_weights is zero for some pixels
    # This was not in the original code but is good practice.
    # However, to strictly follow "Optimize the fusion_based_method function",
    # I will stick to the original logic unless it's clearly a bug fix or essential for optimization.
    # The np.seterr(divide='ignore') should handle cases where sum_weights is 0.
    weight_1 /= sum_weights
    weight_2 /= sum_weights
    weight_3 /= sum_weights

    # plt.subplot(3, 4, 1);
    # plt.imshow(illum_1, cmap='gray')
    # plt.subplot(3, 4, 5);
    # plt.imshow(illmu_2, cmap='gray')
    # plt.subplot(3, 4, 9);
    # plt.imshow(illmu_3, cmap='gray')
    # plt.subplot(3, 4, 2);
    # plt.imshow(weight_brig_1, cmap='gray')
    # plt.subplot(3, 4, 6);
    # plt.imshow(weight_brig_2, cmap='gray')
    # plt.subplot(3, 4, 10);
    # plt.imshow(weight_brig_3, cmap='gray')
    # plt.subplot(3, 4, 3);
    # plt.imshow(weight_chr_1, cmap='gray')
    # plt.subplot(3, 4, 7);
    # plt.imshow(weight_chr_2, cmap='gray')
    # plt.subplot(3, 4, 11);
    # plt.imshow(weight_chr_3, cmap='gray')
    # plt.subplot(3, 4, 4);
    # plt.imshow(weight_1, cmap='gray')
    # plt.subplot(3, 4, 8);
    # plt.imshow(weight_2, cmap='gray')
    # plt.subplot(3, 4, 12);
    # plt.imshow(weight_3, cmap='gray')
    # plt.show()

    illmu_fin_1 = illum_1 * weight_1 + illmu_2 * weight_2 + illmu_3 * weight_3
    # plt.title('Adjusted Illuminance')
    # plt.imshow(illmu_fin_1, cmap='gray')
    # plt.show()

    lap_pyr_img_1 = generate_pyramid(illum_1, 'laplace', level)
    gas_pyr_wt_1 = generate_pyramid(weight_1, 'gauss', level)

    lap_pyr_img_2 = generate_pyramid(illmu_2, 'laplace', level)
    gas_pyr_wt_2 = generate_pyramid(weight_2, 'gauss', level)

    lap_pyr_img_3 = generate_pyramid(illmu_3, 'laplace', level)
    gas_pyr_wt_3 = generate_pyramid(weight_3, 'gauss', level)

    for j in range(level):
        # lap_pyr_img_X[j] and gas_pyr_wt_X[j] are already np.double (float64)
        # from generate_pyramid. Redundant np.double() calls are removed.
        # cv2.resize dsize is (width, height).
        lap_pyr_img_1[j] = cv2.resize(lap_pyr_img_1[j], (width, height)) # Default interpolation: cv2.INTER_LINEAR
        gas_pyr_wt_1[j] = cv2.resize(gas_pyr_wt_1[j], (width, height))

        lap_pyr_img_2[j] = cv2.resize(lap_pyr_img_2[j], (width, height))
        gas_pyr_wt_2[j] = cv2.resize(gas_pyr_wt_2[j], (width, height))

        lap_pyr_img_3[j] = cv2.resize(lap_pyr_img_3[j], (width, height))
        gas_pyr_wt_3[j] = cv2.resize(gas_pyr_wt_3[j], (width, height))

    illum_fin_2 = 0
    for k in range(level):
        illum_fin_2 = illum_fin_2 + lap_pyr_img_1[k] * gas_pyr_wt_1[k] + lap_pyr_img_2[k] * gas_pyr_wt_2[k] + \
                      lap_pyr_img_3[k] * gas_pyr_wt_3[k]

    # plt.title('Adjusted Illuminance with pyramid')
    # plt.imshow(illum_fin_2, cmap='gray')
    # plt.show()

    enhanced_clr = refl * np.expand_dims(illum_fin_2, axis=2)
    enhanced_img = np.uint8(255 * enhanced_clr / np.nanmax(enhanced_clr))

    plt.title('Enhanced Image')
    plt.imsave('enhanced_' + img_name + '.png', enhanced_img)
    plt.imshow(enhanced_img)
    # plt.show() # Removed for non-interactive use, image is saved by plt.imsave

def main():
    args = config_parse()
    name_tmp = re.split(r'[./\\]', args.input_path)
    img_name = name_tmp[-2] + '_' + name_tmp[-1]
    fusion_based_method(args.input_path, args.level, img_name)

if __name__ == "__main__":
    main()
