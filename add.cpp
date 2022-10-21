#include <torch/extension.h>
#include "add.h"
#include <stdio.h>


void torch_lanuch_add2(
  const torch::Tensor & x, const torch::Tensor & y, torch::Tensor & z, int n) {
    // 判断tensor类型
    if (x.device().is_cuda() && y.device().is_cuda() && z.device().is_cuda()) {
      printf("launch with CUDA\n");
      launch_add2_gpu(
        static_cast<const float *>(x.data_ptr()),
        static_cast<const float *>(y.data_ptr()),
        static_cast<float *>(z.data_ptr()),
        n
      );
    } else if (x.device().is_cpu() && y.device().is_cpu() && z.device().is_cpu()) {
      printf("launch with CPU\n");
      launch_add2_cpu(
        static_cast<const float *>(x.data_ptr()),
        static_cast<const float *>(y.data_ptr()),
        static_cast<float *>(z.data_ptr()),
        n
      );
    } else {
      printf("Error! data type must be same.");
    }
  }

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.doc() = "This is my torch custom module.";
  m.def(
    "torch_launch_add2", &torch_lanuch_add2, 
    "this is a torch fun", pybind11::arg("x"), 
    pybind11::arg("y"), pybind11::arg("z"), pybind11::arg("n"));
}

