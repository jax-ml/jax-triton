# Copyright 2023 The jax_triton Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""setup.py for jax-triton."""

import os
import os.path
import posixpath
import shutil
import sys

import setuptools
from setuptools.command import build_ext


class BazelExtension(setuptools.Extension):
  """A C/C++ extension that is defined as a Bazel BUILD target."""

  def __init__(self, bazel_target):
    self.bazel_target = bazel_target
    self.relpath, self.target_name = posixpath.relpath(
        bazel_target, "//"
    ).split(":")
    ext_name = os.path.join(
        self.relpath.replace(posixpath.sep, os.path.sep),
        "jax_triton",
        self.target_name,
    )
    super().__init__(ext_name, sources=[])


class BuildBazelExtension(build_ext.build_ext):
  """A command that runs Bazel to build a C/C++ extension."""

  def run(self):
    for ext in self.extensions:
      self.bazel_build(ext)
    build_ext.build_ext.run(self)

  def bazel_build(self, ext):
    """Runs the bazel build to create the package."""
    if not os.path.exists(self.build_temp):
      os.makedirs(self.build_temp)

    bazel_argv = [
        "bazel",
        "build",
        "--symlink_prefix=" + os.path.join(self.build_temp, "bazel-"),
        "--compilation_mode=" + ("dbg" if self.debug else "opt"),
        "--cxxopt=-std=c++17",
        "--action_env=PYTHON_BIN_PATH=" + sys.executable,
        ext.bazel_target + ".so",
    ]

    self.spawn(bazel_argv)

    ext_bazel_bin_path = os.path.join(
        self.build_temp,
        "bazel-bin",
        ext.relpath,
        ext.target_name + ".so",
    )

    ext_dest_path = self.get_ext_fullpath(ext.name)
    ext_dest_dir = os.path.dirname(ext_dest_path)
    if not os.path.exists(ext_dest_dir):
      os.makedirs(ext_dest_dir)
    shutil.copyfile(ext_bazel_bin_path, ext_dest_path)


setuptools.setup(
    packages=setuptools.find_packages(),
    cmdclass=dict(build_ext=BuildBazelExtension),
    ext_modules=[BazelExtension("//:triton_kernel_call_lib")],
)
