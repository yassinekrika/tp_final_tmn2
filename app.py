import numpy as np
import math
import cv2

# dct compression without using opencv


def dct_cospre_calc(N):
    cospre = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cospre[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N))
    return cospre


def dct_transform(image_data, N):
    # Extract the image dimensions
    height, width = image_data.shape

    # Pre-calculate the DCT matrix using dct_cospre_clac function
    dct_mat = np.zeros((height, width))
    for i in range(height):
        dct_mat = dct_cospre_calc(N)

    # Perform the forward DCT transform on each block of the image
    block_size = 8
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_data[i:i+block_size, j:j+block_size]
            h, w = block.shape
            if (h == 8 and w == 8):
                transformed_block = dct_mat.dot(block).dot(dct_mat.T)
                image_data[i:i+block_size, j:j+block_size] = transformed_block
            else:
                pass

    return image_data

def dct_inverse_transform(transformed_data):
    # Extract the transformed data dimensions
    height, width = transformed_data.shape

    # Create a DCT matrix
    dct_mat = np.zeros((height, width))
    for i in range(height):
        for j in range(width):
            if i == 0:
                dct_mat[i, j] = 1 / np.sqrt(width)
            else:
                dct_mat[i, j] = np.sqrt(2 / width) * np.cos((np.pi / width) * (j + 0.5) * i)

    # Perform the inverse DCT transform on each block of the transformed data
    block_size = 8
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            transformed_block = transformed_data[i:i+block_size, j:j+block_size]
            block = dct_mat.T.dot(transformed_block).dot(dct_mat)
            transformed_data[i:i+block_size, j:j+block_size] = block

    return transformed_data



def psnr(image_true, image_test):
    """Fonction qui calcule le PSNR entre deux images."""
    mse = np.mean((image_true - image_test) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    psnr_val = 20 * math.log10(pixel_max / math.sqrt(mse))
    return psnr_val


if __name__ == '__main__':
    image = cv2.imread('m.bmp', 0)
    # DCT compression
    dct = dct_transform(image, 8)

    idct = dct_inverse_transform(dct)
    # Save image
    dct = np.uint8(dct)

    cv2.imwrite('image_dct.bmp', dct)
    cv2.imwrite('image_idct.bmp', idct)