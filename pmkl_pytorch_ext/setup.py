from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension, CppExtension


setup(
    name='pmkl_pytorch_ext',
    ext_modules=[
        CppExtension(
            'pmkl_pytorch_ext', 
            ['fused_self_attention.cpp'],
        )
    ],
    cmdclass={
        'build_ext': BuildExtension
    },
    packages=['pmkl_function'],
    include_package_data=True,
)
