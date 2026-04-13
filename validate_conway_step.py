import sys

def read_board(filePath):
    with open(filePath, 'rb') as file:
        w = int.from_bytes(file.read(4), byteorder='little')
        h = int.from_bytes(file.read(4), byteorder='little')
        arr = [[0 for x in range(w)] for y in range(h)] 
        for i in range(h):
            for j in range(w):
                arr[i][j] = int.from_bytes(file.read(1), byteorder='little')
    return arr

def count_neighbors(grid, r, c):
    rows = len(grid)
    cols = len(grid[0])
    count = 0
    
    for i in [-1, 0, 1]:
        for j in [-1, 0, 1]:
            if i == 0 and j == 0:
                continue
            newR = r + i
            newC = c + j
            if 0 <= newR < rows and 0 <= newC < cols:
                count += grid[newR][newC]
    return count

def conway(arr):
    m = len(arr)
    n = len(arr[0])

    out = [[0 for _ in range(n)] for _ in range(m)]
    for i in range(m):
        for j in range(n):
            alive_neighbors = count_neighbors(arr, i, j)
            
            if arr[i][j]:
                out[i][j] = int(alive_neighbors == 2 or alive_neighbors == 3)
            else:
                out[i][j] = int(alive_neighbors == 3)
    return out

def main():
    startFile = sys.argv[1]
    arr1 = conway(read_board(startFile))
    stepFile = sys.argv[2]
    arr2 = read_board(stepFile)

    m = len(arr1)
    n = len(arr1[0])
    if m != len(arr2) or n != len(arr2[0]):
        print("Dimensions of boards don't match")
        return 
    for i in range(m):
        for j in range(n):
            if arr1[i][j] != arr2[i][j]:
                print(f"arr1[{i}][{j}] = {arr1[i][j]}, arr2[{i}][{j}] = {arr2[i][j]}")

if __name__ == "__main__":
    main()