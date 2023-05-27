# Pallas, a JAX kernel sublanguage

Pallas allows you to write your own custom kernels using JAX directly!
Some benefits of using Pallas include:

* Provides familiar JAX APIs (`jax.numpy`, `jax.lax`, etc.)
* Compatible with JAX transformations (e.g. `jax.vmap`)

!!! warning
    Pallas is experimental and may not support all JAX ops and transformations! If you find any unexpected errors, please [file an issue on Github](https://github.com/jax-ml/jax-triton/issues/new). Also, Pallas APIs aren't promised to be stable.

Guides:

* [Pallas Quickstart](quickstart.md)
* [Pallas Concepts](concepts.md)
* [`pallas_call`](pallas_call.md)
