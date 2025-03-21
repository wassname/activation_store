from transformers import  PreTrainedModel, PreTrainedTokenizerBase
import torch
from datasets import Dataset
from tqdm.auto import tqdm
import itertools
from torch.utils.data import DataLoader
from loguru import logger
from pathlib import Path
from baukit.nethook import TraceDict, recursive_copy
from einops import rearrange
from datasets.arrow_writer import ParquetWriter
from datasets.fingerprint import Hasher
from transformers.modeling_outputs import ModelOutput
from activation_store.helpers.torch import clear_mem
from typing import Dict, Generator, List, Union, Optional
from torch import Tensor
import tempfile
import inspect

@torch.no_grad()
def default_postprocess_result(input: dict, trace: TraceDict, output: ModelOutput, model: PreTrainedModel, act_groups=Optional[Dict[str,List[str]]], last_token=True, dtype=torch.float16, save_hidden_states=True) -> Dict[str, Tensor]:
    """Make your own. This adds activations to output, and rearranges hidden states.

    Note the parquet write support float16, so we use that. It does not support float8, bfloat16, etc.

    Often you only want the last token, which makes things smaller and easily stackable.
    
    """
    token_index = slice(-1, None) if last_token else slice(None)

    # Baukit records the literal layer output, which varies by model. Sometimes you get a tuple, or not.Usually [b, t, h] for MLP, but not for attention layers. You may need to customize this.
    if act_groups is not None:
        acts = {}
        for grp_nm, group in act_groups.items():
            grp_acts = [v.output[0] if isinstance(v.output, tuple) else v.output for lyr_nm, v in trace.items() if lyr_nm in group]
            assert len(grp_acts) > 0, f"no activations found for {group}"
            assert grp_acts[0].dim() == 3, f"expected [b, t, h] activations, got {grp_acts[0].shape}"
            grp_acts = torch.stack([a[:, token_index] for a in grp_acts], dim=1)
            acts[f'acts-{grp_nm}'] = grp_acts.to(dtype)
    else:
        acts = {f'acts-{lyr_nm}': 
                v.output[0] if isinstance(v.output, tuple) else v.output
                for lyr_nm, v in trace.items()}
        acts = {k: v[:, token_index].to(dtype) for k, v in acts.items() if v is not None}
    del trace
    
    # batch must be first, also the writer supports float16 so lets use that
    if save_hidden_states and output.hidden_states is not None:
        output.hidden_states = rearrange([h[:, token_index] for h in output.hidden_states], 'l b t h -> b l t h').to(dtype)
    else:
        output.hidden_states = None

    output.logits = output.logits[:, token_index].to(dtype)

    o = dict(**acts, **output)
    if ('attention_mask' in input) and not last_token:
        o['attention_mask'] = input['attention_mask']
    if 'label' in input:
        o['label'] = input['label']

    # all output tensors must have a batch dim
    for grp_nm, v in o.items():
        if v.dim() == 0:
            bs = input['input_ids'].shape[0]
            o[grp_nm] = v.repeat(bs)


    # finally check for nans
    for grp_nm, v in o.items():
        if torch.isnan(v).any():
            raise ValueError(f"nan found in {grp_nm}")
    return o


@torch.no_grad()
def generate_batches(loader: DataLoader, model: PreTrainedModel, layers = [], postprocess_result=default_postprocess_result) -> Generator[Dict[str, Tensor], None, None]:
    """
    Collect activations from a model

    Args:
    - loader: DataLoader
    - model: PreTrainedModel
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
        # with torch.autocast(device_type=device.type):

        # FIXME for some reason autocast isn't converting the inputs
        batch = {k: v.to(device) if isinstance(v, Tensor) else v for k, v in batch.items()}
        if layers is not None:
            with TraceDict(model, layers, retain_grad=False, detach=True) as trace:
                out = model(**batch, use_cache=False, output_hidden_states=True, return_dict=True)
        else:
            out = model(**batch, use_cache=False, output_hidden_states=True, return_dict=True)
            trace = None
        o = postprocess_result(batch, trace, out, model, act_groups=act_groups)

        # copy to avoid memory leaks
        o = {k: v.detach().cpu().contiguous().clone() if isinstance(v, Tensor) else v for k, v in o.items()}
        o = recursive_copy(o)
        out = trace = batch = None
        clear_mem()
        yield o


def output_dataset_hash(**kwargs):
    # special hash for some key parts that don't hash well
    func = generate_batches
    name = "%s.%s" % (func.__module__, func.__name__)
    kwargs['func'] = name
    
    for k,v in kwargs.items():
        if isinstance(v, Dataset):
            kwargs[k] = f"Dataset_{v._fingerprint}_{len(v)}"
        elif isinstance(v, DataLoader):
            kwargs[k] = f"DataLoader.dataset_{v.dataset._fingerprint}_{len(v)}_{v.batch_size}"
        elif isinstance(v, PreTrainedTokenizerBase):
            raise NotImplementedError("hashing tokenizers not implemented")
        elif isinstance(v, PreTrainedModel):
            kwargs[k] = f"PreTrainedModel_{v.config._name_or_path}" # PretrainedConfig
        elif inspect.isfunction(v):
            kwargs[k] = "Function: %s.%s" % (v.__module__, v.__name__)
    logger.debug(f"hashing {kwargs}")
    suffix = Hasher.hash(kwargs)
    return suffix


def activation_store(loader: DataLoader, model: PreTrainedModel, dataset_name='', layers: Union[List[str], Dict[str, List[str]]]=[], dataset_dir=None, writer_batch_size=1, postprocess_result=default_postprocess_result, outfile: Optional[Path] = None) -> Dataset:
    """
    Collect activations from a model and store them in a dataset

    Args:
    - loader: DataLoader
    - model: PreTrainedModel
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
    # FIXME I think this is the problem, instead of using a naive hash I will need to custom hash some key parts, model_name, dataset_name, layers, etc
    hash = output_dataset_hash(generate_batches=generate_batches, loader=loader, model=model, layers=layers, postprocess_result=postprocess_result)

    if outfile is None:
        outdir = Path(tempfile.mkdtemp(prefix='activation_store'))
        outfile = outdir / f"ds_act_{dataset_name}_{hash}.parquet"
    outfile.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"creating dataset {outfile}")

    iterator = generate_batches(loader, model, layers=layers, postprocess_result=postprocess_result) 

    with ParquetWriter(path=outfile, writer_batch_size=writer_batch_size,
                       embed_local_files=True
                       ) as writer: 
        for bo in iterator:

            bs = len(next(iter(bo.values())))
            assert all(len(v) == bs for v in bo.values()), "must return Dict[str,Tensor] and all tensors with same batch size as first dimension"

            # or maybe better compression to `writer.write(example, key)` for each
            writer.write_batch(bo)
            del bo
            clear_mem()
        writer.finalize() 
        writer.close()
    
    return outfile
