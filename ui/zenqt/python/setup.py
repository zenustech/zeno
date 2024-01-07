from setuptools import Extension, setup
setup(ext_modules=[Extension("custom", ["zenopyapi.cpp"])])