import numpy.typing as npt
from .config import Config
import torch
import numpy as np
def get_batch(dataset: npt.NDArray,config:Config):
    cond=len(dataset)-config.max_seq_len
    indexs=torch.randint(cond,size=(config.batch_size,))
    X=np.stack([dataset[index:index+config.max_seq_len] for index in indexs])
    Y=np.stack([dataset[index+1:index+config.max_seq_len+1] for index in indexs])
    
    X=torch.tensor(X,dtype=torch.int64,device=config.device)
    Y=torch.tensor(Y,dtype=torch.int64,device=config.device)
    return (X,Y)
