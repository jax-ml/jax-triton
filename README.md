# jax-triton

The `jax-triton` repository contains integrations between [JAX](https://github.com/google/jax) and [Triton](https://github.com/openai/triton).

*This is not an officially supported Google product.*

## Installation

```bash
$ pip install jax-triton
```

Make sure you have a CUDA-compatible `jaxlib` installed.
For example you could run:
```bash
$ pip install "jax[cuda11_cudnn82]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
```

## Development

To develop `jax-triton`, you can clone the repo with:
```bash
$ git clone https://github.com/jax-ml/jax-triton.git 
```
and do an editable install with:
```bash
$ cd jax-triton
$ pip install -e .
```
To run the `jax-triton` tests, you'll need `pytest`:
```bash
$ pip install pytest
$ pytest tests/
```
