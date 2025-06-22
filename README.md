# activation_store

Utility library to persistently store transformer activations on disk.

These activations can be quite large (layers x batch x sequence x hidden_size), so generating them to disk helps avoid out of memory errors.

Install using 
```
pip install git+https://github.com/wassname/activation_store.git
```

Example
```py
layer_groups = {'mlp.down_proj': [
  'model.layers.21.mlp.down_proj',
  'model.layers.22.mlp.down_proj',
  'model.layers.23.mlp.down_proj'],
 'self_attn': [
  'model.layers.21.self_attn',
  'model.layers.22.self_attn',
  'model.layers.23.self_attn'],
 'mlp.up_proj': [
  'model.layers.21.mlp.up_proj',
  'model.layers.22.mlp.up_proj',
  'model.layers.23.mlp.up_proj']}

# collect activations into a huggingface dataset
f = activation_store(ds, model, layers=layer_groups)
f
# > Generating train split: 0 examples [00:00, ? examples/s]
# Dataset({
#    features: ['mlp.down_proj', 'self_attn', 'mlp.up_proj', 'loss', 'logits', 'hidden_states'],
#    num_rows: 20
# })

# it has this sgaoe
ds_a = Dataset.from_parquet(str(f)).with_format("torch")
ds_a[0:2]['hidden_states'].shape # [batch, layers, tokens, hidden_states]
# torch.Size([2, 25, 1, 896])

```


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
