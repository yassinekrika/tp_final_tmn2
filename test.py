import numpy as np
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
    frequency = freq
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

    # print(huffmanCode)
    return huffmanCode, frequency  

def encode_rle(message):
    encoded_string = []
    i = 0
    while i < len(message):
        if message[i] != 0:
            # Concatenate non-zero values until a 0 is encountered
            j = i
            while j < len(message) and message[j] != 0:
                j += 1
            encoded_string.append(''.join(str(x) for x in message[i:j]))
            i = j
        else:
            # Count consecutive zeros and add to output
            count = 1
            j = i + 1
            while j < len(message) and message[j] == 0:
                count += 1
                j += 1
            encoded_string.append('#' + str(count))
            i = j
    return encoded_string

bit_stream_null = encode_rle([0, 0, 0, 0, 3, 4, 3, 0, 0, 0, 0, 6, 5, 0]) 
print(bit_stream_null)

def reverse_decode_rle(encoded_message):
    decoded_string = []
    for code in encoded_message:
        if code[0] == '#':
            # Decode consecutive zeros
            count = int(code[1:])
            decoded_string += [0] * count
        else:
            # Decode concatenated non-zero values
            decoded_string += [int(x) for x in code]
    return decoded_string

reverse = reverse_decode_rle(['#4', '343', '#4', '65', '#1']) 
print(reverse)

# '000000bbbbb00000bbbb0000ccc' => 216 39 82% only huffman
# '#5bbbbb#4bbbb#4ccc' => 144 60 58% rle + huffman