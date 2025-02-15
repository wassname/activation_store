# cache_transformer_activations

Utility library to collect transformer activations **on disk**.

These activations can be quite large (layers x batch x sequence x hidden_size), so it's nice to store it on disk and avoid and out of memory error.

Install using `pip install git+https://github.com/wassname/cache_transformer_activations.git`.

## Development
```
git clone https//github.com/wassname/cache_transformer_activations.git
uv sync
```

see exampes in `nbs` folder.
