from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from setuptools import setup
import os
import torch


def get_moudules():
    sources = ["add.cpp", "add_cpu.cpp"]
    now_dir = os.path.dirname(os.path.abspath(__file__))
    include_dirs = [now_dir]
    moudules = []
    if torch.cuda.is_available():
        sources += ["add_cuda.cu"]
        moudules.append(
            CUDAExtension(
                name="_ext_add",
                sources=sources,
                include_dirs=include_dirs,
                # with_cuda=True,
                # define_macros=[("WITH_CUDA", None)]
            )
        )
    else:
        moudules.append(
            CppExtension(
                name="_ext_add",
                sources=sources,
                include_dirs=include_dirs
            )
        )
    return moudules
        

def get_includes():
    return []


setup(
    name="add2",
    version="0.0.1",
    description="this is add demo pytorch plugin",
    author="Tlntin",
    author_email="TlntinDeng01@Gmail.com",
    requires=["torch"],
    ext_modules=get_moudules(),
    include_dirs=get_includes(),
    cmdclass={"build_ext": BuildExtension}
)
