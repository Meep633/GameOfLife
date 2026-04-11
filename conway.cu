#include<stdio.h>
#include<cuda.h>
#include<cuda_runtime.h>

extern "C" 
{
int setCudaDevice(int rank);
void runCuda(int device, int *x, int *y, size_t w, size_t h);
void cudaUnifiedMalloc(int** x, size_t nBytes);
void cudaUnifiedFree(void* x);
}

__global__ void Conway(int *x, int *y, size_t w, size_t h);

int setCudaDevice(int rank) {
  int numCudaDevices;
  cudaGetDeviceCount(&numCudaDevices);
  int device = rank % numCudaDevices; 
  cudaSetDevice(device);
  return device;
}

void runCuda(int device, int *x, int *y, size_t w, size_t h)
{
  cudaMemPrefetchAsync(x - haloSz, ((2 * haloSz) + n) * sizeof(int), device, 0);
  cudaMemPrefetchAsync(y, n * sizeof(int), device, 0);
  size_t sharedMemSize = (1024 + 2 * haloSz) * sizeof(int);
  Conway<<<16384,1024, sharedMemSize>>>(x, y, w, h);
  cudaDeviceSynchronize();
}

void cudaUnifiedMalloc(bool** x, size_t nBytes) {
  cudaMallocManaged(x, nBytes);
}

void cudaUnifiedFree(void* x) {
  cudaFree(x);
}

__global__
void Conway(int *x, int *y, size_t w, size_t h)
{
  extern __shared__ int s_data[];

  int stride = blockDim.x * gridDim.x;

  for (int base = blockIdx.x * blockDim.x; base < n; base += stride) {
    int global_idx = base + threadIdx.x;
    if(global_idx < n + haloSz) {
      s_data[haloSz + threadIdx.x] = x[global_idx];
    } else {
      s_data[haloSz + threadIdx.x] = 0;
    }

    for (int h = threadIdx.x; h < haloSz; h += blockDim.x) {
      s_data[h] = x[base + h - haloSz]; 
    }

    for (int h = threadIdx.x; h < haloSz; h += blockDim.x) {
        int right_idx = base + blockDim.x + h;
        if(right_idx < n + haloSz) {
          s_data[haloSz + blockDim.x + h] = x[right_idx];
        } else {
          s_data[haloSz + blockDim.x + h] = 0;
        }
    }

    __syncthreads();

    if (global_idx < n) {
        int result = 0;
        for (int j = 0; j <= 2 * haloSz; ++j) {
            result += s_data[threadIdx.x + j];
        }
        y[global_idx] = result;
    }

    __syncthreads(); 
  }
}
