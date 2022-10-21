"""
使用load_inline加载扩展
适用于比较短的函数
"""
from torch.utils.cpp_extension import load_inline
import time
import torch
import os


cpu_sources = [
"""
void launch_add2_cpu(const float * x, const float * y, float * z, int n) {
  for (int i = 0; i < n; ++i) {
    z[i] = x[i] + y[i];
  }
}
""",

"""
#include <stdio.h>
#include "add.h"

void torch_launch_add2(
  const torch::Tensor & x, const torch::Tensor & y, torch::Tensor & z, int n) {
    // 判断tensor类型
    if (x.device().is_cuda() && y.device().is_cuda() && z.device().is_cuda()) {
      printf("launch with CUDA\\n");
      launch_add2_gpu(
        static_cast<const float *>(x.data_ptr()),
        static_cast<const float *>(y.data_ptr()),
        static_cast<float *>(z.data_ptr()),
        n
      );
    } else if (x.device().is_cpu() && y.device().is_cpu() && z.device().is_cpu()) {
      printf("launch with CPU\\n");
      launch_add2_cpu(
        static_cast<const float *>(x.data_ptr()),
        static_cast<const float *>(y.data_ptr()),
        static_cast<float *>(z.data_ptr()),
        n
      );
    }
  }
""",
]

cuda_sources = [
"""
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
"""
]

now_dir = os.path.dirname(os.path.abspath(__file__))

if torch.cuda.is_available():
    inline_module = load_inline(
        name="inline_extend", cpp_sources=cpu_sources,
        cuda_sources=cuda_sources,
        functions=["torch_launch_add2"],
        extra_include_paths=[now_dir]
    )
else:
    inline_module = load_inline(
        name="inline_extend", cpp_sources=cpu_sources,
        functions=["torch_launch_add2"],
        extra_include_paths=[now_dir]
    )


def run(use_cuda=True):
    if use_cuda and torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    n = 1024000
    x = torch.ones([n]).to(device)
    y = torch.rand([n]).to(device)
    print("x: ", x[:5])
    print("y: ", y[:5])
    z = torch.zeros([n]).to(device)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    st = time.time()
    inline_module.torch_launch_add2(x, y, z, n)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    et = time.time()
    print("z", z[:5])
    print("during ", et - st)


if __name__ == "__main__":
    run(use_cuda=True)
    print("=" * 20)
    run(use_cuda=False)
