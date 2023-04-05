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

http_archive(
    name = "bazel_skylib",
    sha256 = "b8a1527901774180afc798aeb28c4634bdccf19c4d98e7bdd1ce79d1fe9aaad7",
    urls = [
        "https://mirror.bazel.build/github.com/bazelbuild/bazel-skylib/releases/download/1.4.1/bazel-skylib-1.4.1.tar.gz",
        "https://github.com/bazelbuild/bazel-skylib/releases/download/1.4.1/bazel-skylib-1.4.1.tar.gz",
    ],
)

http_archive(
  name = "com_google_absl",
  sha256 = "9a2b5752d7bfade0bdeee2701de17c9480620f8b237e1964c1b9967c75374906",
  strip_prefix = "abseil-cpp-20230125.2",
  urls = ["https://github.com/abseil/abseil-cpp/archive/20230125.2.tar.gz"],
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

git_repository(
  name = "pybind11_abseil",
  remote = "https://github.com/pybind/pybind11_abseil.git",
  branch = "master",
)
