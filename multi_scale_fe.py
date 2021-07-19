import re
import cv2
import math
import argparse
import numpy as np
import scipy.signal
from PIL import Image
import matplotlib.pyplot as plt

def config_parse():
    parser = argparse.ArgumentParser(description='Path Parser')
    parser.add_argument('--input_path', type=str, required=True, help='Specify the path of the input image.')
    parser.add_argument('--level', type=int, default=3, help='Specify the number of layers of the multi-scale pyramid.')
    args = parser.parse_args()

    return args

def box_filter(img, rad):
    [height, width] = img.shape
    im_dst = np.zeros_like(img)

    im_cum = np.cumsum(img, 0)
    im_dst[0: rad + 1, :] = im_cum[rad: 2 * rad + 1, :]
    im_dst[rad + 1: height - rad, :] = im_cum[2 * rad + 1: height, :] - im_cum[0: height - 2 * rad - 1, :]
    im_dst[height - rad: height, :] = np.tile(im_cum[height - 1, :], [rad, 1]) - im_cum[height - 2 * rad - 1: height - rad - 1, :]

    im_cum = np.cumsum(im_dst, 1)
    im_dst[:, 0: rad + 1] = im_cum[:, rad: 2 * rad + 1]
    im_dst[:, rad + 1: width - rad] = im_cum[:, 2 * rad + 1: width] - im_cum[:, 0: width - 2 * rad - 1]
    im_dst[:, width - rad: width] = np.tile(im_cum[:, width - 1], [rad, 1]).T - im_cum[:, width - 2 * rad - 1: width - rad - 1]

    return im_dst

def guided_filter(guide, img, rad, eps):
    [height, weight] = guide.shape
    n = box_filter(np.ones((height, weight)), rad)

    guide_mean = box_filter(guide, rad) / n
    img_mean = box_filter(img, rad) / n
    guid_img_mean = box_filter(guide * img, rad) / n
    guid_img_cov = guid_img_mean - guide_mean * img_mean

    guid2_mean = box_filter(guide * guide, rad) / n
    guide_var = guid2_mean - guide_mean * guide_mean

    a = guid_img_cov / (guide_var + eps)
    b = img_mean - a * guide_mean

    a_mean = box_filter(a, rad) / n
    b_mean = box_filter(b, rad) / n

    guide_img = a_mean * guide + b_mean

    return guide_img

def pyramid_reduce(img):
    kernel_width = 5
    center_width = 0.375
    kernel_1d = np.array([0.25 - center_width / 2, 0.25, center_width, 0.25, 0.25 - center_width / 2])
    kernel = np.kron(kernel_1d, kernel_1d.T).reshape(len(kernel_1d), len(kernel_1d))

    img = np.double(img)
    size = np.array(img.shape)
    img_out = 0

    img_filtered = cv2.filter2D(img.astype(np.float32), -1, kernel, borderType=cv2.BORDER_REPLICATE)
    img_out = img_out + img_filtered[:size[0]:2, :size[1]:2]

    return img_out

def pyramid_expand(img):
    kernel_width = 5
    center_width = 0.375
    kernel_1d = np.array([0.25 - center_width / 2, 0.25, center_width, 0.25, 0.25 - center_width / 2])
    kernel = np.kron(kernel_1d, kernel_1d.T).reshape(len(kernel_1d), len(kernel_1d)) * 4

    kernel_00 = kernel[:kernel_width:2, :kernel_width:2]
    kernel_01 = kernel[:kernel_width:2, 1:kernel_width:2]
    kernel_10 = kernel[1:kernel_width:2, :kernel_width:2]
    kernel_11 = kernel[1:kernel_width:2, 1:kernel_width:2]

    img = np.double(img)
    size = np.array(img.shape)
    out_size = size * 2 - 1
    img_out = np.zeros(out_size)

    img_ph = np.pad(img, ((0, 0), (1, 1)), 'edge')
    img_pv = np.pad(img, ((1, 1), (0, 0)), 'edge')

    img_00 = cv2.filter2D(img.astype(np.float32), -1, kernel_00, borderType=cv2.BORDER_REPLICATE)
    img_01 = scipy.signal.convolve2d(img_pv, kernel_01, 'valid')
    img_10 = scipy.signal.convolve2d(img_ph, kernel_10, 'valid')
    img_11 = scipy.signal.convolve2d(img, kernel_11, 'valid')

    img_out[:out_size[0]:2, :out_size[1]:2] = img_00
    img_out[1:out_size[0]:2, :out_size[1]:2] = img_10
    img_out[:out_size[0]:2, 1:out_size[1]:2] = img_01
    img_out[1:out_size[0]:2, 1:out_size[1]:2] = img_11

    return img_out

def generate_pyramid(img, type, level):
    pyramid = [0 for i in range(level)]
    pyramid[0] = np.double(img)
    for i in range(1, level):
        pyramid[i] = pyramid_reduce(pyramid[i - 1])

    if type == 'gauss':
        return pyramid

    for j in range(level):
        out_size = np.array(pyramid[level - 1 - j].shape) * 2 - 1
        pyramid[level - 2 - j] = pyramid[level - 2 - j][:out_size[0], :out_size[1]]

    for k in range(level - 1):
        pyramid[k] = pyramid[k] - pyramid_expand(pyramid[k + 1])

    return pyramid

def fusion_based_method(img_path, level, img_name):
    img_load = Image.open(img_path)
    plt.title('Original Image')
    plt.imshow(img_load)
    plt.show()

    img = np.double(np.array(img_load))
    red_chan = img[:, :, 0];
    green_chan = img[:, :, 1];
    blue_chan = img[:, :, 2]
    [height, width, depth] = img.shape

    lightness = np.maximum(red_chan, np.maximum(green_chan, blue_chan))
    # plt.title('Image Lightness')
    # plt.imshow(lightness, cmap='gray')
    # plt.show()

    struct_elem = np.zeros((15, 15))
    n = 4
    for i in range(1, struct_elem.shape[0] - 1):
        n = n - 1 if np.abs(i - int(struct_elem.shape[1] / 2)) > 3 else 0
        for j in range(np.abs(n) + 1, struct_elem.shape[1] - np.abs(n) - 1):
            struct_elem[i, j] = 1
    struct_elem[0, int(struct_elem.shape[1] / 2)] = 1;
    struct_elem[-1, int(struct_elem.shape[1] / 2)] = 1;
    struct_elem[int(struct_elem.shape[0] / 2), 0] = 1;
    struct_elem[int(struct_elem.shape[0] / 2), -1] = 1
    struct_elem = np.uint8(struct_elem)

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

    illmu_2 = 2 / math.pi * np.arctan(lamb * illum)

    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8, 8))
    illmu_3 = clahe.apply(np.uint8(illum * 255)) / 255

    alpha = 2; phi = 250

    weight_brig_1 = np.exp(-(illum_1 - 0.5) ** 2 / 0.125)
    weight_brig_2 = np.exp(-(illmu_2 - 0.5) ** 2 / 0.125)
    weight_brig_3 = np.exp(-(illmu_3 - 0.5) ** 2 / 0.125)

    weight_chr_1 = illum_1 * (1 + np.cos(alpha * hsv_img[:, :, 0] + phi) * hsv_img[:, :, 1])
    weight_chr_2 = illmu_2 * (1 + np.cos(alpha * hsv_img[:, :, 0] + phi) * hsv_img[:, :, 1])
    weight_chr_3 = illmu_3 * (1 + np.cos(alpha * hsv_img[:, :, 0] + phi) * hsv_img[:, :, 1])

    weight_1 = weight_brig_1 * weight_chr_1;
    weight_2 = weight_brig_2 * weight_chr_2;
    weight_3 = weight_brig_3 * weight_chr_3
    weight_1 /= (weight_1 + weight_2 + weight_3);
    weight_2 /= (weight_1 + weight_2 + weight_3);
    weight_3 /= (weight_1 + weight_2 + weight_3)

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
        lap_pyr_img_1[j] = cv2.resize(np.double(lap_pyr_img_1[j]), [width, height])
        gas_pyr_wt_1[j] = cv2.resize(np.double(gas_pyr_wt_1[j]), [width, height])

        lap_pyr_img_2[j] = cv2.resize(np.double(lap_pyr_img_2[j]), [width, height])
        gas_pyr_wt_2[j] = cv2.resize(np.double(gas_pyr_wt_2[j]), [width, height])

        lap_pyr_img_3[j] = cv2.resize(np.double(lap_pyr_img_3[j]), [width, height])
        gas_pyr_wt_3[j] = cv2.resize(np.double(gas_pyr_wt_3[j]), [width, height])

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
    plt.show()

def main():
    args = config_parse()
    name_tmp = re.split('[./\\\]', args.input_path)
    img_name = name_tmp[-2] + '_' + name_tmp[-1]
    fusion_based_method(args.input_path, args.level, img_name)

if __name__ == "__main__":
    main()