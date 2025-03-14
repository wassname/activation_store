from transformers import AutoModelForCausalLM
import torch
from datasets import Dataset
from tqdm.auto import tqdm
import itertools
from torch.utils.data import DataLoader
from loguru import logger
from pathlib import Path
from baukit.nethook import TraceDict, recursive_copy
from einops import rearrange
from datasets.arrow_writer import ArrowWriter, ParquetWriter
from datasets.fingerprint import Hasher
from transformers.modeling_outputs import ModelOutput
from activation_store.helpers.torch import clear_mem
from typing import Dict, Generator, List, Union, Optional
from torch import Tensor
import os

default_output_folder = (Path(__file__).parent.parent / "outputs").resolve()

@torch.no_grad()
def default_postprocess_result(input: dict, trace: TraceDict, output: ModelOutput, model: AutoModelForCausalLM, act_groups=Optional[Dict[str,List[str]]], last_token=True, dtype=torch.float16) -> Dict[str, Tensor]:
    """Make your own. This adds activations to output, and rearranges hidden states.

    Note the parquet write support float16, so we use that. It does not support float8, bfloat16, etc.

    Often you only want the last token, which makes things smaller and easily stackable.
    
    """
    token_index = slice(-1, None) if last_token else slice(None)

    # Baukit records the literal layer output, which varies by model. Sometimes you get a tuple, or not.Usually [b, t, h] for MLP, but not for attention layers. You may need to customize this.
    if act_groups is not None:
        acts = {}
        for k, group in act_groups.items():
            aas = [v.output[0] if isinstance(v.output, tuple) else v.output for k, v in trace.items() if k in group]
            assert len(aas) > 0, f"no activations found for {group}"
            aas = torch.stack([a[:, token_index].to(dtype) for a in aas], dim=1)
            acts[k] = aas
    else:
        acts = {f'act-{k}': 
                v.output[0][:, token_index].to(dtype) if isinstance(v.output, tuple) else v.output[:, token_index]
                for k, v in trace.items()}
    del trace
    
    # batch must be first, also the writer supports float16 so lets use that
    output.hidden_states = rearrange(list(output.hidden_states), 'l b t h -> b l t h')[:, :, token_index].to(dtype)

    output.logits = output.logits[:, token_index].to(dtype)

    o = dict(**acts, **output)
    if ('attention_mask' in input) and not last_token:
        o['attention_mask'] = input['attention_mask']
    if 'label' in input:
        o['label'] = input['label']

    # all output tensors must have a batch dim
    for k, v in o.items():
        if v.dim() == 0:
            bs = input['input_ids'].shape[0]
            o[k] = v.repeat(bs)
    return o


@torch.no_grad
def generate_batches(loader: DataLoader, model: AutoModelForCausalLM, layers = [], postprocess_result=default_postprocess_result) -> Generator[Dict[str, Tensor], None, None]:
    """
    Collect activations from a model

    Args:
    - loader: DataLoader
    - model: AutoModelForCausalLM
    - layers: can be
        - selected from `model.named_modules()`
        - groups of layers to collect, these will be stacked so they must have compatible sizes
    - postprocess_result: Callable - see `default_postprocess_result` for signature

    Returns:
    - Generator of [Dict[str, Tensor]], where each tensor has shape [batch,...]
    """
    act_groups = None
    if isinstance(layers, dict):
        act_groups = layers
        layers = list(itertools.chain(*layers.values()))
    
    model.eval()
    for batch in tqdm(loader, 'collecting activations'):
        device = next(model.parameters()).device
        with torch.autocast(device_type=device.type):

            # FIXME for some reason autocast isn't converting the inputs
            batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
            with TraceDict(model, layers, retain_grad=False, detach=True, clone=True) as trace:
                out = model(**batch, use_cache=False, output_hidden_states=True, return_dict=True)
        o = postprocess_result(batch, trace, out, model, act_groups=act_groups)

        # copy to avoid memory leaks
        o = {k: v.to('cpu') if isinstance(v, Tensor) else v for k, v in o.items()}
        o = recursive_copy(o)
        out = trace = batch = None
        clear_mem()
        yield o


def dataset_hash(**kwargs):
    suffix = Hasher.hash(kwargs)
    return suffix


def activation_store(loader: DataLoader, model: AutoModelForCausalLM, dataset_name='', layers: Union[List[str], Dict[str, List[str]]]=[], dataset_dir=default_output_folder, writer_batch_size=1, postprocess_result=default_postprocess_result) -> Dataset:
    """
    Collect activations from a model and store them in a dataset

    Args:
    - loader: DataLoader
    - model: AutoModelForCausalLM
    - dataset_name: str
    - layers: 
        - List[str]  selected from `model.named_modules()`
        - or Dict[str, List[str]]] - groups of layers to collect, these will be stacked so they must have compatible sizes
    - dataset_dir: Path
    - postprocess_result: Callable - see `default_postprocess_result` for signature

    Returns:
    - file

    Usage:
        f = activation_store(loader, model, layers=['transformer.h'])
        Dataset.from_parquet(f).with_format("torch")
    """
    hash = dataset_hash(generate_batches=generate_batches, loader=loader, model=model)
    f = dataset_dir / ".ds" / f"ds_{dataset_name}_{hash}.parquet"
    f.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"creating dataset {f}")

    iterator = generate_batches(loader, model, layers=layers, postprocess_result=postprocess_result) 

    with ParquetWriter(path=f, writer_batch_size=writer_batch_size,
                       embed_local_files=True
                       ) as writer: 
        for bo in iterator:

            bs = len(next(iter(bo.values())))
            assert all(len(v) == bs for v in bo.values()), "must return Dict[str,Tensor] and all tensors with same batch size as first dimension"

            # or maybe better compression to `writer.write(example, key)` for each
            writer.write_batch(bo)
        writer.finalize() 
        writer.close()
    
    return f
