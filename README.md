# activation_store

Utility library to persistently store transformer activations on disk.

These activations can be quite large (layers x batch x sequence x hidden_size), so generating them to disk helps avoid out of memory errors.

Install using `pip install git+https://github.com/wassname/activation_store.git`.

## Development
```
git clone https//github.com/wassname/activation_store.git
uv sync
```

see exampes in `nbs` folder.
