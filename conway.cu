#include <stdio.h>
#include <stdbool.h>
#include <cuda.h>
#include <cuda_runtime.h>

#define BLOCK_SIZE (16)

extern "C" 
{
int setCudaDevice(int rank);
void runCuda(int device, bool *x, bool *y, size_t w, size_t h);
void cudaUnifiedMalloc(bool** x, size_t nBytes);
void cudaUnifiedFree(void* x);
}

__global__ void Conway(bool *x, bool *y, size_t w, size_t h);

int setCudaDevice(int rank) {
  int numCudaDevices;
  cudaGetDeviceCount(&numCudaDevices);
  int device = rank % numCudaDevices; 
  cudaSetDevice(device);
  return device;
}

void runCuda(int device, bool *x, bool *y, size_t w, size_t h)
{
  cudaMemPrefetchAsync(x, (w + 2) * (h + 2) * sizeof(bool), device, 0);
  cudaMemPrefetchAsync(y, (w + 2) * (h + 2) * sizeof(bool), device, 0);

  int threadsPerBlock = (BLOCK_SIZE * BLOCK_SIZE);
  int gridX = (w + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int gridY = (h + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int totalBlocks = gridX * gridY;

  Conway<<<totalBlocks, threadsPerBlock>>>(x, y, w, h);
  cudaDeviceSynchronize();
}

void cudaUnifiedMalloc(bool** x, size_t nBytes) {
  cudaMallocManaged(x, nBytes);
}

void cudaUnifiedFree(void* x) {
  cudaFree(x);
}

__global__
void Conway(bool *in, bool *out, size_t w, size_t h)
{
  __shared__ bool tile[BLOCK_SIZE + 2][BLOCK_SIZE + 2];

  int tx = threadIdx.x % BLOCK_SIZE;
  int ty = threadIdx.x / BLOCK_SIZE;

  int blocksPerRow = (w + BLOCK_SIZE - 1) / BLOCK_SIZE;
  int bx = blockIdx.x % blocksPerRow;
  int by = blockIdx.x / blocksPerRow;

  int gx = (bx * BLOCK_SIZE) + tx + 1;
  int gy = (by * BLOCK_SIZE) + ty + 1;

  int stride = w + 2;

  bool right = (tx == BLOCK_SIZE - 1 || gx == w);
  bool bottom = (ty == BLOCK_SIZE - 1 || gy == h);

  if (gx <= w && gy <= h) {
    tile[ty + 1][tx + 1] = in[gy * stride + gx];
  } else {
    tile[ty + 1][tx + 1] = 0;
  }

  if (tx == 0 && gy <= h) {
    tile[ty + 1][0] = in[gy * stride + (gx - 1)];
  }
  if (right && gy <= h && gx <= w) {
    tile[ty + 1][tx + 2] = in[gy * stride + (gx + 1)];
  }
  if (ty == 0 && gx <= w) {
    tile[0][tx + 1] = in[(gy - 1) * stride + gx];
  }
  if (bottom && gx <= w && gy <= h) {
    tile[ty + 2][tx + 1] = in[(gy + 1) * stride + gx];
  }

  if (tx == 0 && ty == 0) { 
    tile[0][0] = in[(gy - 1) * stride + (gx - 1)];
  }
  if (right && ty == 0 && gx <= w) {
    tile[0][tx + 2] = in[(gy - 1) * stride + (gx + 1)];
  }
  if (tx == 0 && bottom && gy <= h) {
    tile[ty + 2][0] = in[(gy + 1) * stride + (gx - 1)];
  }
  if (right && bottom && gx <= w && gy <= h) {
    tile[ty + 2][tx + 2] = in[(gy + 1) * stride + (gx + 1)];
  }

  __syncthreads();

  if (gx <= w && gy <= h) {
    int aliveNeighbors = tile[ty][tx] + 
      tile[ty][tx + 1] + 
      tile[ty][tx + 2] +
      tile[ty + 1][tx] + 
      tile[ty + 1][tx + 2] +
      tile[ty + 2][tx] + 
      tile[ty + 2][tx + 1] + 
      tile[ty + 2][tx + 2];

    bool current_state = tile[ty + 1][tx + 1];
    
    out[gy * stride + gx] = (current_state && (aliveNeighbors == 2 || aliveNeighbors == 3)) || (!current_state && aliveNeighbors == 3);
  }  
}