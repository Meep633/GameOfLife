#include <stdio.h>
#include <stdlib.h>
#include <fcntl.h>
#include <stdbool.h>
#include <unistd.h>
#include <string.h>

typedef unsigned long long ticks;

// IBM POWER9 System clock with 512MHZ resolution.
static __inline__ ticks getticks(void)
{
  unsigned int tbl, tbu0, tbu1;

  do {
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu0));
    __asm__ __volatile__ ("mftb %0" : "=r"(tbl));
    __asm__ __volatile__ ("mftbu %0" : "=r"(tbu1));
  } while (tbu0 != tbu1);

  return (((unsigned long long)tbu0) << 32) | tbl;
}

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
    bool writeSteps = atoi(argv[4]);
    
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

    ticks start = getticks();
    for(int i = 0; i < steps; ++i) {
        conway(arr, swap, m, n);
        if(writeSteps) {
            writeFile(swap, m, n, outputDir, i);
        }
        bool** tmp = swap;
        swap = arr;
        arr = tmp;
    }
    ticks end = getticks();
    double time = (double)(end - start) / (double)512000000.0;
    printf("Total time to compute %d steps of Conways game of life is %lf seconds \n", steps, time);
    if(!writeSteps) {
        writeFile(arr, m, n, outputDir, steps);
    }
    for(int i = 0; i < m; ++i) {
        free(arr[i]);
        free(swap[i]);
    }
    free(arr);
    free(swap);

    return 0;
}