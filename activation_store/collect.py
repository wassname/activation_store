from transformers import AutoModelForCausalLM
import torch
from datasets import Dataset
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from loguru import logger
from pathlib import Path
from baukit.nethook import TraceDict, recursive_copy
from einops import rearrange
from datasets.arrow_writer import ArrowWriter 
from datasets.fingerprint import Hasher
from transformers.modeling_outputs import ModelOutput

from activation_store.helpers.torch import clear_mem
from typing import Dict, Generator
from torch import Tensor

default_output_folder = (Path(__file__).parent.parent.parent / "outputs").resolve()

def default_postprocess_result(input: dict, ret: TraceDict, output: ModelOutput) -> Dict[str, Tensor]:
    """add ret, activations to output"""

    # Baukit records the literal layer output, which varies by model. Here we assume that the output or the first part are activations we want
    acts = {f'act-{k}': 
            v.output[0] if isinstance(v.output, tuple) else v.output
            for k, v in ret.items()}
    
    output.hidden_states = rearrange(list(output.hidden_states), 'l b t h -> b l t h')

    return dict(**acts, **output)


@torch.no_grad
def generate_batches(loader: DataLoader, model: AutoModelForCausalLM, layers, postprocess_result=default_postprocess_result) -> Generator[Dict[str, Tensor], None, None]:
    model.eval()
    for batch in tqdm(loader, 'collecting hidden states'):
        device = next(model.parameters()).device
        b_in = {
            k: v.to(device)
            for k, v in batch.items()
        }
        with TraceDict(model, layers) as ret:
            out = model(**b_in, use_cache=False, output_hidden_states=True, return_dict=True)
        o = postprocess_result(batch, ret, out)
        o = recursive_copy(o)
        out = ret = b_in = None
        clear_mem()
        yield o


def dataset_hash(**kwargs):
    suffix = Hasher.hash(kwargs)
    return suffix


def activation_store(loader: DataLoader, model: AutoModelForCausalLM, dataset_name='', layers=[], dataset_dir=default_output_folder, writer_batch_size=1, postprocess_result=default_postprocess_result) -> Dataset:
    """
    Collect activations from a model and store them in a dataset

    Args:
    - loader: DataLoader
    - model: AutoModelForCausalLM
    - dataset_name: str
    - layers: List[str] - selected from `model.named_modules()`
    - dataset_dir: Path
    - postprocess_result: Callable - see `default_postprocess_result` for signature
    """
    hash = dataset_hash(generate_batches=generate_batches, loader=loader, model=model)
    f = dataset_dir / ".ds" / f"ds_{dataset_name}_{hash}"
    f.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"creating dataset {f}")

    iterator = generate_batches(loader, model, layers=layers, postprocess_result=postprocess_result)
    with ArrowWriter(path=f, writer_batch_size=writer_batch_size) as writer: 
        for bo in iterator:

            bs = len(next(iter(bo.values())))
            assert all(len(v) == bs for v in bo.values()), f"must return Dict[str,Tensor] and all tensors with same batch size a first dimension"

            writer.write_batch(bo)
        writer.write_examples_on_file()
        writer.finalize() 
    
    ds = Dataset.from_file(str(f)).with_format("torch")
    return ds, f
