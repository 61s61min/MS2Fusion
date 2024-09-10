# file: setup.py
from distutils.core import setup
from Cython.Build import cythonize

setup(name='common',
      ext_modules=cythonize("common.py"))