import numpy.typing as npt
from .config import Config
import torch
import numpy as np
def get_batch(dataset: npt.NDArray,config:Config):
    tokens=torch.tensor(dataset,dtype=torch.int64,device=config.device)
    cond=len(tokens)-config.max_seq_len
    indexs=torch.randint(cond,size=(config.batch_size,))
    X=torch.stack([tokens[index:index+config.max_seq_len] for index in indexs])
    Y=torch.stack([tokens[index+1:index+config.max_seq_len+1] for index in indexs])

    return (X,Y)

