import os
from distutils.core import setup, Extension
import distutils.sysconfig
import cudaSetup

def main():
    ext = cudaSetup.createCUDAExtension(name="mPyCUDA", sources=['src/module.cu'])
    setup(name="mPyCUDA",
          version="0",
          description="Python interface for some custom CUDA functions",
          author="Minh Pham",
          ext_modules=[ext],
          cmdclass={'build_ext': cudaSetup.custom_build_ext},
          include_dirs=['include/', 'src/'],
          zip_safe=False
          )

if __name__ == "__main__":
    main()