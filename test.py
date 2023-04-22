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

def hufman_coding(zigza_value):

    string = zigza_value

    def huffman_code_tree(node):
        codes = {}

        def traverse(node, code):
            if type(node) is int:
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
    print(freq)
    nodes = freq

    while len(nodes) > 1:
        (key1, c1) = nodes[-1]
        (key2, c2) = nodes[-2]
        nodes = nodes[:-2]
        node = NodeTree(key1, key2)
        nodes.append((node, c1 + c2))

        nodes = sorted(nodes, key=lambda x: x[1], reverse=True)

    huffmanCode = huffman_code_tree(nodes[0][0])

    return huffmanCode

arr = [0, 0, 1, 2, 4, 0]

hufman_coding(arr)