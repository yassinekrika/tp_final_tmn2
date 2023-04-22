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
        else:
            encoded_string = encoded_string + str(ch)
        i = j + 1
    return encoded_string

val = [0,0,0,0,0,0,0,0,0,2,0,0,0,0,0,3,4,50,0,0]

print(encode_message(val))