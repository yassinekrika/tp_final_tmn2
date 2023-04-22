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

    dct_mat = dct_cospre_calc(N)
    img= np.zeros((height, width))

    # Perform the forward DCT transform on each block of the image
    block_size = 8
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_data[i:i+block_size, j:j+block_size]
            block = np.float32(block)
            h, w = block.shape
            if (h == 8 and w == 8):
                transformed_block = np.dot(np.dot(dct_mat, block), np.transpose(dct_mat))
                img[i:i+block_size, j:j+block_size] = transformed_block
            else:
                pass
    return img

def idct_transform(image_data, N):
    # Extract the image dimensions
    height, width = image_data.shape

    idct_mat = dct_cospre_calc(N)
    img= np.zeros((height, width))
    # Perform the inverse DCT transform on each block of the image
    block_size = 8
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_data[i:i+block_size, j:j+block_size]
            
            h, w = block.shape
            if (h == 8 and w == 8):
                transformed_block = np.dot(np.dot(np.transpose(idct_mat), block), idct_mat)
                img[i:i+block_size, j:j+block_size] = transformed_block
            else:
                pass
    print(img)
    return np.uint8(img)

def psnr(image_true, image_test):
    """Fonction qui calcule le PSNR entre deux images."""
    mse = np.mean((image_true - image_test) ** 2)
    if mse == 0:
        return float('inf')
    pixel_max = 255.0
    psnr_val = 20 * math.log10(pixel_max / math.sqrt(mse))
    return psnr_val


if __name__ == '__main__':
    image = cv2.imread('m.bmp')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # DCT compression
    dct = dct_transform(gray, 8)

    idct = idct_transform(dct, 8)
    # Save image
    dct = np.uint8(dct)

    cv2.imwrite('image_dct.bmp', dct)
    cv2.imwrite('image_idct.bmp', idct)