# activation_store

Utility library to persistently store transformer activations on disk.

These activations can be quite large (layers x batch x sequence x hidden_size), so generating them to disk helps avoid out of memory errors.

Install using `pip install git+https://github.com/wassname/activation_store.git`.

## Development
```
git clone https//github.com/wassname/activation_store.git
uv sync
```

see examples in `nbs` folder.


## TODO:

- [x] test compression: it's not worth the [complexity](https://github.com/EleutherAI/elk/blob/84e99a36a5050881d85f1510a2486ce46ac1f942/elk/extraction/extraction.py#L382)
- [x] add examples
- [ ] generate and collect activations
  - A manual loop of forwards/generate, reusing kv_cache, and appending model outputs along the token dim. saving outputs too
