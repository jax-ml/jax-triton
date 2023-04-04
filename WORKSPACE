workspace(name = "jax_triton")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

new_local_repository(
  name = "cuda",
  path = "/usr/local/cuda",
  build_file_content = """
cc_library(
    name = "cuda_headers",
    hdrs = glob(["include/*.h"]),
    includes = ["include/"],
    visibility = ["//visibility:public"],
)

cc_library(
    name = "libcuda",
    srcs = ["lib64/stubs/libcuda.so"],
    visibility = ["//visibility:public"],
)
  """
)

git_repository(
  name = "pybind11_bazel",
  remote = "https://github.com/pybind/pybind11_bazel.git",
  branch = "master",
)

http_archive(
  name = "pybind11",
  build_file = "@pybind11_bazel//:pybind11.BUILD",
  strip_prefix = "pybind11-2.10.4",
  sha256 = "832e2f309c57da9c1e6d4542dedd34b24e4192ecb4d62f6f4866a737454c9970",
  urls = ["https://github.com/pybind/pybind11/archive/v2.10.4.tar.gz"],
)

load("@pybind11_bazel//:python_configure.bzl", "python_configure")
python_configure(name = "local_config_python")
