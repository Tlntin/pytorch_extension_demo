from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension
from setuptools import setup, find_packages
import os
import torch
from glob import glob


def get_moudules():
    sources = glob("add2/src/*.cpp")
    now_dir = os.path.dirname(os.path.abspath(__file__))
    src_dir = os.path.join(now_dir, "add2", "src")
    include_dirs = [src_dir]
    moudules = []
    if torch.cuda.is_available():
        sources += glob("add2/src/*.cu")
        sources = [os.path.join(now_dir, file) for file in sources]
        print("sources", sources)
        moudules.append(
            CUDAExtension(
                name="add2._ext_add",
                sources=sources,
                include_dirs=include_dirs,
                # with_cuda=True,
                # define_macros=[("WITH_CUDA", None)]
            )
        )
    else:
        moudules.append(
            CppExtension(
                name="add2._ext_add",
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
    packages=find_packages(exclude=["test", "setup.py"]),
    requires=["torch"],
    ext_modules=get_moudules(),
    include_dirs=get_includes(),
    cmdclass={"build_ext": BuildExtension}
)
