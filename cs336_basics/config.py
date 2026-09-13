from dataclasses import dataclass
import torch

@dataclass
class Config:
    # model
    d_model:int=4096
    num_layers:int=16
    vocab_size:int=10000

    # attention
    num_heads: int=16
    max_seq_len: int=1024
    theta: float=10000


    # FFN
    d_ff: int=float(8/3*d_model)

    # RMSNorm
    eps:float=1e-5

    # train
    batch_size:int=64

    # device
    device:str='cuda' if torch.cuda.is_available() else 'cpu'

    # AdamW
    lr:float=2e-4
    betas:tuple[float,float]=(0.9,0.99)
    weight_decay:float=0.9

    # lr_schedule
    max_learning_rate:float=lr
    min_learning_rate:float=lr/10.0
    warmp_iters:int=500
    cosine_cycle_iters:int=10000
    
    