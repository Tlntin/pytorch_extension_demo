"""
使用setup.py将扩展安装到Python环境，然后进行导包并开始使用
支持CPU和GPU两种方式启动，可以通过use_cuda测试cpu和cuda启动效果
"""

import torch
import time
import _ext_add


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
    _ext_add.torch_launch_add2(x, y, z, n)
    if use_cuda and torch.cuda.is_available():
        torch.cuda.synchronize()
    et = time.time()
    print("z", z[:5])
    print("during ", et - st)


if __name__ == "__main__":
    print("=" * 20)
    print("正式运行")
    run(use_cuda=True)
    print("=" * 20)
    run(use_cuda=False)