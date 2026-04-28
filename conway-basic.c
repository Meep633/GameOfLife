#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>

int count_neighbors(bool** arr, int m, int n, int i, int j) {
    int cnt = 0;
    for (int r = i - 1; r <= i + 1; ++r) {
        for (int c = j - 1; c <= j + 1; ++c) {
            if (r == i && c == j) {
                continue;
            }

            if (r >= 0 && r < m && c >= 0 && c < n && arr[r][c]) {
                ++cnt;
            }
        }
    }
    return cnt;
}

void conway(bool** arr, bool** swap, int m, int n) {
    for(int i = 0; i < m; ++i) {
        for(int j = 0; j < n; ++j) {
            int alive = count_neighbors(arr, m, n, i, j);
            
            if(arr[i][j]) {
                swap[i][j] = (alive == 2 || alive == 3);
            } else {
                swap[i][j] = (alive == 3);
            }
        }
    }
}

void writeFile(bool** arr, int m, int n, char* outputDir, int step) {
    char outputFileName[strlen(outputDir) + 17];
    snprintf(outputFileName, sizeof(outputFileName), "%s/step_%d", outputDir, step);

    int fd = open(outputFileName, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if(fd == -1) {
        perror("Error creating output file");
        return ;
    }
    int rc = write(fd, &m, sizeof(int));
    rc = write(fd, &n, sizeof(int));
    for(int i = 0; i < m; ++i) {
        rc = write(fd, arr[i], n * sizeof(bool));
    }

    close(fd);
}

int main(int argc, char** argv) {
    char* inputFile = argv[1];
    char* outputDir = argv[2];
    int steps = atoi(argv[3]);
    
    int fd = open(inputFile, O_RDONLY);
    if(fd == -1) {
        return EXIT_FAILURE;
    }

    int m;
    int n;

    int rc = read(fd, &m, sizeof(int));
    rc = read(fd, &n, sizeof(int));
    bool** arr = malloc(m * sizeof(bool*));
    bool** swap = malloc(m * sizeof(bool*));
    for(int i = 0; i < m; ++i) {
        arr[i] = malloc(n * sizeof(bool));
        swap[i] = malloc(n * sizeof(bool));

        rc = read(fd, arr[i], n * sizeof(bool));
    }

    close(fd);

    for(int i = 0; i < steps; ++i) {
        conway(arr, swap, m, n);

        writeFile(swap, m, n, outputDir, i);
        bool** tmp = swap;
        swap = arr;
        arr = tmp;
    }
    return 0;
}