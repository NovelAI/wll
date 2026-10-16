# The extension module; the package metadata is in pyproject.toml.
import glob

from setuptools import Extension, setup

setup(
    ext_modules=[
        Extension(
            "wll._wll",
            sources=["python/_wll.c"] + sorted(glob.glob("src/*.c")),
            include_dirs=["src"],
            # The encoder needs GCC or Clang (builtins, target attributes).
            extra_compile_args=["-O3", "-std=c11"],
        )
    ]
)
