from setuptools import Extension, setup
from Cython.Build import cythonize
import numpy as np

setup(
    ext_modules = cythonize([
        Extension("transition_probability",
                  sources = ["transition_probability.pyx"],
                  libraries = ["m"],
                  include_dirs = [np.get_include()],
                  extra_compile_args=['-O3'],
                  define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')]
                  )],
                            language_level = 3)
)
