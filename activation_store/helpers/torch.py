import torch
import gc


def clear_mem():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.synchronize()  
        torch.cuda.empty_cache()
        gc.collect()
