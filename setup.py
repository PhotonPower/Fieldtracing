"""The optional C++ extension fieldtracing._mesh (mesh edges, two-ring and the quadratic fit of
fieldtracing.mesh); everything else is in pyproject.toml. Optional: without a compiler the
package installs without it and uses the Python code, which stays the reference."""

from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

mesh = Pybind11Extension("fieldtracing._mesh", ["cpp/mesh_module.cpp"], cxx_std=17)
mesh.optional = True

setup(ext_modules=[mesh], cmdclass={"build_ext": build_ext})
