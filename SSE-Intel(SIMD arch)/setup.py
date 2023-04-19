from distutils.core import setup, Extension

setup(name="assignment", ext_modules=[
    Extension("assignment", ["assignment.c"])
])
