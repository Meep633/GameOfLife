import sys
import random

def main():
    w = int(sys.argv[1])
    h = int(sys.argv[2])
    onePercent = float(sys.argv[3])
    filePath = sys.argv[4]

    with open(filePath, 'wb') as file:
        file.write(w.to_bytes(4, byteorder='little'))
        file.write(h.to_bytes(4, byteorder='little'))
        numElements = w * h
        for i in range(numElements):
            value = 1 if random.random() < onePercent else 0
            file.write(value.to_bytes(1, byteorder='little'))

if __name__ == "__main__":
    main()