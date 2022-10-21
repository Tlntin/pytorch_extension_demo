#include "add.h"
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>


__global__ void add2_kernel(const float * x, const float * y, float * z, int n) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n) {
    z[idx] = x[idx] + y[idx];
  }
}

void launch_add2_gpu(const float * x, const float * y, float * z, int n) {
  dim3 block;
  if (n > 1024) {
    block.x = 1024;
  } else if (n > 512) {
    block.x = 512;
  } else if (n > 256) {
    block.x = 256;
  } else if (n > 128) {
    block.x = 128;
  } else {
    block.x = 8;
  }
  dim3 grid;
  grid.x = (n + block.x - 1) / block.x;
  add2_kernel<<<grid, block>>>(x, y, z, n);
}