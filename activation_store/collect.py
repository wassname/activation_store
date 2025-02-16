import gc
from pathlib import Path
from typing import Dict, Generator
import copy
import torch
from baukit.nethook import TraceDict, recursive_copy
from datasets import Dataset
from datasets.arrow_writer import ParquetWriter
from datasets.fingerprint import Hasher
from einops import rearrange
from loguru import logger
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM
from transformers.modeling_outputs import ModelOutput

default_output_folder = (Path(__file__).parent.parent / "outputs").resolve()


def clear_mem():
    gc.collect()
    torch.cuda.empty_cache()
    gc.collect()


def default_postprocess_result(
    input: dict, trace: TraceDict, output: ModelOutput, model: AutoModelForCausalLM
) -> Dict[str, Tensor]:
    """add activations to output, and rearrange hidden states"""

    # Baukit records the literal layer output, which varies by model. Here we assume that the output or the first part are activations we want
    acts = {
        f"act-{k}": v.output[0] if isinstance(v.output, tuple) else v.output
        for k, v in trace.items()
    }

    output.hidden_states = rearrange(list(output.hidden_states), "l b t h -> b l t h")

    o = dict(
        attention_mask=input["attention_mask"],
        **acts, **output
    )
    return o

def to_cpu(x):
    """
    Trys to convert torch if possible a single item
    """
    if isinstance(x, torch.Tensor):
        x = x.cpu()
        return x
    else:
        return x

def recursive_copy2(x, clone=None, detach=None, retain_grad=None):
    """
    from baukit with addition of deep copy for non tensors
    
    Copies a reference to a tensor, or an object that contains tensors,
    optionally detaching and cloning the tensor(s).  If retain_grad is
    true, the original tensors are marked to have grads retained.
    """
    if not clone and not detach and not retain_grad:
        return x
    if isinstance(x, torch.Tensor):
        if retain_grad:
            if not x.requires_grad:
                x.requires_grad = True
            x.retain_grad()
        elif detach:
            x = x.detach()
        if clone:
            x = x.clone()
        return x
    # Only dicts, lists, and tuples (and subclasses) can be copied.
    if isinstance(x, dict):
        return type(x)({k: recursive_copy(v, clone=clone, detach=detach, retain_grad=retain_grad) for k, v in x.items()})
    elif isinstance(x, (list, tuple)):
        return type(x)([recursive_copy(v, clone=clone, detach=detach, retain_grad=retain_grad) for v in x])
    else:
        return copy.deepcopy(x)

@torch.no_grad
def generate_batches(
    loader: DataLoader,
    model: AutoModelForCausalLM,
    layers,
    postprocess_result=default_postprocess_result,
) -> Generator[Dict[str, Tensor], None, None]:
    model.eval()
    for batch in tqdm(loader, "collecting activations"):
        device = next(model.parameters()).device
        with torch.amp.autocast(device_type=device.type):
            with TraceDict(model, layers) as trace:
                out = model(
                    **batch,
                    use_cache=False,
                    output_hidden_states=True,
                    return_dict=True,
                )
        o = postprocess_result(batch, trace, out, model)

        # copy to avoid memory leaks
        for k in o:
            if not isinstance(o[k], torch.Tensor):
                print('o', k, type(o[k]))
        o = {k: to_cpu(v) for k, v in o.items()}
        o = recursive_copy(o, clone=True, detach=True)

        from datasets.features.features import cast_to_python_objects
        o = cast_to_python_objects(o, only_1d_for_numpy=False, optimize_list_casting=False)
        out = trace = batch = None
        clear_mem()
        yield o


def dataset_hash(**kwargs):
    suffix = Hasher.hash(kwargs)
    return suffix


def activation_store(
    loader: DataLoader,
    model: AutoModelForCausalLM,
    dataset_name="",
    layers=[],
    dataset_dir=default_output_folder,
    writer_batch_size=1,
    postprocess_result=default_postprocess_result,
    features=None,
    schema=None,
) -> Dataset:
    """
    Collect activations from a model and store them in a dataset

    Args:
    - loader: DataLoader
    - model: AutoModelForCausalLM
    - dataset_name: str
    - layers: List[str] - selected from `model.named_modules()`
    - dataset_dir: Path
    - postprocess_result: Callable - see `default_postprocess_result` for signature

    Returns:
    - file

    Usage:
        f = activation_store(loader, model, layers=['transformer.h'])
        Dataset.from_parquet(f)
    """
    hash = dataset_hash(generate_batches=generate_batches, loader=loader, model=model)
    f = dataset_dir / ".ds" / f"ds_{dataset_name}_{hash}.parquet"
    f.parent.mkdir(exist_ok=True, parents=True)
    logger.info(f"creating dataset {f}")

    iterator = generate_batches(
        loader, model, layers=layers, postprocess_result=postprocess_result
    )

    # batch_1 = next(iterator)
    # Features.encode_batch(batch_1)
    # features = Features({'x': Array2D(shape=(1, 3), dtype='int32')})

    with ParquetWriter(
        path=f, writer_batch_size=writer_batch_size, embed_local_files=True,
         features=features, schema=schema,
    ) as writer:
        # writer.write_batch(batch_1)
        for bo in iterator:
            bs = len(next(iter(bo.values())))
            assert all(len(v) == bs for v in bo.values()), (
                "must return Dict[str,Tensor] and all tensors with same batch size a first dimension"
            )
            writer.write_batch(bo)
        writer.finalize()
        writer.close()

    # ds = Dataset.from_file(str(f)).with_format("torch")
    return f
