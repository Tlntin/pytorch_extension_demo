"""
使用load方式加载扩展
"""
import torch
from torch.utils.cpp_extension import load
import os
import time


use_cuda = True  # 是否开启cuda，用于测试cpu与GPU运行效果
now_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(now_dir)
src_dir = os.path.join(parent_dir, "add2", "src")
file_list = os.listdir(src_dir)

if torch.cuda.is_available():
    file_list = [
        file for file in file_list
        if file.endswith(".cpp") or file.endswith(".cu")
    ]
else:
    file_list = [
        file for file in file_list
        if file.endswith(".cpp")
    ]
sources = [os.path.join(src_dir, file) for file in file_list]


add_module = load(
    name="_ext_add",
    sources=sources,
    extra_include_paths=[src_dir]
)


def run(use_cuda=True):
    if use_cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
    add_module.torch_launch_add2(x, y, z, n)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    et = time.time()
    print("z", z[:5])
    print("during ", et - st)


if __name__ == "__main__":
    run(use_cuda=True)
    print("=" * 20)
    run(use_cuda=False)
