import numpy as np
import cv2

# dct compression without using opencv
def dct_cospre_calc(N):
    cospre = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            cospre[i, j] = np.cos(np.pi * i * (2 * j + 1) / (2 * N))
    return cospre

#  Implantez l’algorithme qui effectue la transformation par bloc d’une image (dct), puis la transformation par bloc inverse (dct_inv).
#  Vous devez utiliser la fonction dct_cospre_calc pour calculer les cosinus précalculés.
#  Vous devez utiliser la fonction cv2.dct pour calculer la transformation par bloc d’une image.
#  Vous devez utiliser la fonction cv2.idct pour calculer la transformation par bloc inverse d’une image.
def dct(image, N):
    # dct compression
    dct = np.zeros(image.shape)
    cospre = dct_cospre_calc(N)
    for i in range(0, image.shape[0], N):
        for j in range(0, image.shape[1], N):
            dct[i:i+N, j:j+N] = cospre.dot(image[i:i+N, j:j+N]).dot(cospre.T)

    return dct

def main():
    # Read image
    image = cv2.imread('m.bmp', 0)
    # DCT compression
    dct = dct(image, 8)
    # Save image
    dct = np.uint8(dct) 
    cv2.imwrite('image_idct.bmp', dct)

if __name__ == '__main__':  
    main()