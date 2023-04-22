import numpy as np
import cv2
import math

img = cv2.imread('m.bmp', cv2.IMREAD_GRAYSCALE)
img_original = cv2.imread('m.bmp', cv2.IMREAD_GRAYSCALE)


def dct_transform(image_data, N):
    # Extract the image dimensions
    height, width = image_data.shape

    img= np.zeros((height, width))

    # Perform the forward DCT transform on each block of the image
    block_size = 8
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_data[i:i+block_size, j:j+block_size]
            block = np.float32(block)
            h, w = block.shape
            if (h == 8 and w == 8):
                transformed_block = cv2.dct(block)
                img[i:i+block_size, j:j+block_size] = transformed_block
                
            else:
                pass

    return img

def idct_transform(image_data, N):
    # Extract the image dimensions
    height, width = image_data.shape

    img= np.zeros((height, width))

    # Perform the forward DCT transform on each block of the image
    block_size = 8
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_data[i:i+block_size, j:j+block_size]

            h, w = block.shape
            if (h == 8 and w == 8):
                transformed_block = cv2.idct(block)
                img[i:i+block_size, j:j+block_size] = transformed_block
            else:
                pass
    return img

# img1_32 = np.float32(img)

# dct = cv2.dct(img1_32)
dct = dct_transform(img, 8)

# img2 = cv2.idct(dct)
img2 = idct_transform(dct, 8)

img2 = np.uint8(img)

cv2.imwrite('img.jpeg', img)
cv2.imwrite('img_dct.jpeg', dct)
cv2.imwrite('img_idct.jpeg', img2)

def psnr(original_img, reconstructed_img):
    m, n = img_original.shape
    some = 0
    r = 255
    for i in range(1, m-1):
        for j in range(1, n-1):
            some = some + (original_img[i, j] - reconstructed_img[i, j]) ** 2

    pnsr = 10 * math.log10(r ** 2 / (some / (m * n)))
    print('PSNR value is :', pnsr)

psnr(img, img2)

def appl_masque(image_data, masque, index):
    height, width = image_data.shape
    img= np.zeros((height, width))

    # Perform the forward DCT transform on each block of the image
    block_size = 8
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_data[i:i+block_size, j:j+block_size]
            block = np.float32(block)
            h, w = block.shape
            if (h == 8 and w == 8):
                transformed_block = cv2.dct(block)

                for i in range(len(masque[0])):
                    for j in range(len(masque[1])):
                        transformed_block[i][j] = masque[i][j] * transformed_block[i][j]
                
                img[i:i+block_size, j:j+block_size] = transformed_block
            else:
                pass

    print(f'Masque '+str(index)+' ***********************************')
    psnr(img_original, img)


masque1= [[1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0]]

masque2= [[1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 1],
          [1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0]]

masque3= [[1, 1, 1, 1, 1, 1, 1, 0],
          [1, 1, 1, 1, 1, 1, 0, 0],
          [1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]

masque4= [[1, 1, 1, 1, 1, 0, 0, 0],
          [1, 1, 1, 1, 0, 0, 0, 0],
          [1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]

masque5= [[1, 1, 1, 0, 0, 0, 0, 0],
          [1, 1, 0, 0, 0, 0, 0, 0],
          [1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]

masque6= [[1, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0],
          [0, 0, 0, 0, 0, 0, 0, 0]]

appl_masque(img, masque1, 1)
appl_masque(img, masque2, 2)
appl_masque(img, masque3, 3)
appl_masque(img, masque4, 4)
appl_masque(img, masque5, 5)
appl_masque(img, masque6, 6)

def quantdct(image_data, quant, alpha):
    height, width = image_data.shape
    img = np.zeros((height, width))

    # Perform the forward DCT transform on each block of the image
    block_size = 8
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_data[i:i+block_size, j:j+block_size]
            block = np.float32(block)
            h, w = block.shape
            if (h == 8 and w == 8):
                transformed_block = cv2.dct(block)

                transformed_block = np.around(transformed_block / (quant * alpha))
                
                img[i:i+block_size, j:j+block_size] = transformed_block
            else:
                pass
    return img

def iquantdct(image_data, quant):
    height, width = image_data.shape
    img = np.zeros((height, width))

    # Perform the forward DCT transform on each block of the image
    block_size = 8
    for i in range(0, height, block_size):
        for j in range(0, width, block_size):
            block = image_data[i:i+block_size, j:j+block_size]

            h, w = block.shape
            if (h == 8 and w == 8):

                for i in range(len(quant[0])):
                    for j in range(len(quant[1])):
                        block[i][j] = block[i][j] * quant[i][j]

                img[i:i+block_size, j:j+block_size] = block
            else:
                pass
    return img

quant= [[16, 11, 10, 16, 24, 40, 51, 61],
          [12, 12, 14, 19, 26, 58, 60, 55],
          [14, 13, 16, 24, 40, 57, 69, 56],
          [14, 17, 22, 29, 51, 87, 80, 62],
          [18, 22, 37, 56, 68, 109, 103, 77],
          [24, 35, 55, 64, 81, 104, 113, 92],
          [49, 64, 78, 87, 103, 121, 120, 101],
          [72, 92, 95, 98, 112, 100, 103, 99]]

img_qunt = quantdct(img_original, quant, alpha=1)
print('block quantife : ***************************************')
print(img_qunt[0:8, 0:8])

img_iqunt = iquantdct(img_qunt, quant)
print('block iquantife : ***************************************')
print(img_iqunt[0:8, 0:8])

def zigzag(matrice):
    l,c=matrice.shape
    i = 0
    j = 0
    ret = 0
    li=[]
    li.append(matrice[i][j])
    while(len(li)<c*l):
        if(j<c-1):
            j=j+1
        else:
            i=i+1
        while (i<l and j>=0):
            li.append(matrice[i][j])
            i+=1
            j-=1
        j+=1
        i-=1   
        if (i<l-1):
            i+=1
        else:
            j+=1
        while (j<c and i>=0):
            li.append(matrice[i][j])
            i-=1
            j+=1
        j-=1
        i+=1
    return li 

zigza_value = zigzag(img_qunt)
print(zigza_value[1])


# huffman coding
class NodeTree(object):

    def __init__(self, left=None, right=None):
        self.left = left
        self.right = right

    def children(self):
        return (self.left, self.right)

    def nodes(self):
        return (self.left, self.right)

    def __str__(self):
        return '%s_%s' % (self.left, self.right)

def huffman_coding(zigza_value):

    string = zigza_value

    def huffman_code_tree(node):
        codes = {}

        def traverse(node, code):
            if type(node) is int or type(node) is np.float64 or type(node) is str:
                codes[node] = code
            else:
                traverse(node.left, code + '0')
                traverse(node.right, code + '1')

        traverse(node, '')
        return codes

    # Calculating frequency
    freq = {}
    for c in string:
        if c in freq:
            freq[c] += 1
        else:
            freq[c] = 1

    freq = sorted(freq.items(), key=lambda x: x[1], reverse=True)
    
    nodes = freq

    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))

        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

    huffmanCode = huffman_code_tree(nodes[0][0])

    print(huffmanCode)
    return huffmanCode

string = 'asdf;lakjfioquwrlksajdfoijz'
print('Huffman coding : ***************************************')
huffman_coding(string)

print('Huffman coding not null values : ***************************************')
not_null_val = []
for val in zigza_value:
    if(val != 0.0):
        not_null_val.append(val)

huffman_coding(not_null_val)

print('Coding null values : ***************************************')
def encode_message(message):
    encoded_string = ""
    i = 0
    while (i <= len(message)-1):
        count = 1
        ch = message[i]
        j = i
        while (j < len(message)-1):    
            if (message[j] == message[j + 1]): 
                count = count + 1
                j = j + 1
            else: 
                break
        if (ch == 0.0):
            encoded_string = encoded_string + str(count) + '#'
        i = j + 1
    return encoded_string

print(encode_message(zigza_value))