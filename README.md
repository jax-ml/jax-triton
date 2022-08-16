# jax-triton

The `jax-triton` repository contains integrations between [JAX](https://github.com/google/jax) and [Triton](https://github.com/openai/triton).

*This is not an officially supported Google product.*

## Installation

```bash
$ pip install jax-triton
```

## Development

To develop `jax-triton`, you can clone the repo with:
```bash
$ git clone https://github.com/jax-ml/jax-triton.git 
```
and do an editable install with:
```bash
$ pip install -e jax-triton
```
To run the `jax-triton` tests, you'll need `pytest`:
```bash
$ pip install pytest
$ pytest tests/
```
