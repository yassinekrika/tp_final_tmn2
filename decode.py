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


def huffman_decode(string, codes):
    decoded = ""
    code = ""
    for char in string:
        code += char
        for k, v in codes.items():
            if v == code:
                decoded += k
                code = ""
                break
    return decoded

huffman_code, freq = huffman_coding('yassine')
only_bit_stream = ''
for i in 'yassine':
    if (i in huffman_code):
        only_bit_stream += huffman_code[i]
    else:
        only_bit_stream += i
print(huffman_code)
print(only_bit_stream)

huffman_decode = huffman_decode(only_bit_stream, huffman_code)
print(huffman_decode)