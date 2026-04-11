import sys

def main():
    filePath = sys.argv[1]

    with open(filePath, 'rb') as file:
        w = int.from_bytes(file.read(4), byteorder='little')
        h = int.from_bytes(file.read(4), byteorder='little')
        print(f"Printing {w}x{h} board")
        for i in range(h):
            for j in range(w):
                print(int.from_bytes(file.read(1), byteorder='little'), end=' ')
            print()

if __name__ == "__main__":
    main()