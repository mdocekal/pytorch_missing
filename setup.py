# -*- coding: UTF-8 -*-
"""
Created on 05.11.21

:author:     Martin Dočekal
"""
from setuptools import setup, find_packages
import torch
from torch.utils import cpp_extension


def is_requirement(line):
    return not (line.strip() == "" or line.strip().startswith("#"))


with open('README.md') as readme_file:
    README = readme_file.read()

with open("requirements.txt") as f:
    REQUIREMENTS = [line.strip() for line in f if is_requirement(line)]

ext_modules = [
    cpp_extension.CppExtension('pytorch_missing.indices_dot_product', ['pytorch_missing/csrc/indices_dot_product.cpp'], extra_compile_args=['-O3']),
    cpp_extension.CppExtension('pytorch_missing.indices_scatter', ['pytorch_missing/csrc/indices_scatter.cpp'], extra_compile_args=['-O3']),
    cpp_extension.CppExtension('pytorch_missing.ranges', ['pytorch_missing/csrc/ranges.cpp'], extra_compile_args=['-O3'])
]

if torch.cuda.is_available():
    ext_modules.append(cpp_extension.CUDAExtension('pytorch_missing.indices_dot_product_cuda', ['pytorch_missing/csrc/cuda/indices_dot_product.cu'], extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}))
    ext_modules.append(cpp_extension.CUDAExtension('pytorch_missing.indices_scatter_cuda', ['pytorch_missing/csrc/cuda/indices_scatter.cu'], extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}))
    ext_modules.append(cpp_extension.CUDAExtension('pytorch_missing.ranges_cuda', ['pytorch_missing/csrc/cuda/ranges.cu'], extra_compile_args={'cxx': ['-O3'], 'nvcc': ['-O3']}))

setup_args = dict(
    name='pytorch_missing',
    version='1.0.0',
    description='Package with missing features in PyTorch.',
    long_description_content_type="text/markdown",
    long_description=README,
    license='The Unlicense',
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    author='Martin Dočekal',
    keywords=['torch'],
    url='https://github.com/mdocekal/pytorch_missing',
    python_requires='>=3.10',
    install_requires=REQUIREMENTS,
    ext_modules=ext_modules,
    cmdclass={'build_ext': cpp_extension.BuildExtension.with_options(no_python_abi_suffix=True, use_ninja=False)}
)

if __name__ == '__main__':
    setup(**setup_args)
