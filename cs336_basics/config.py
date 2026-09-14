from dataclasses import dataclass
import torch

@dataclass
class Config:
    # model
    d_model:int=512
    num_layers:int=4
    vocab_size:int=10000

    # attention
    num_heads: int=16
    max_seq_len: int=256
    theta: float=10000


    # FFN
    d_ff: int=1344

    # RMSNorm
    eps:float=1e-5

    # device
    device:str='cuda' if torch.cuda.is_available() else 'cpu'

    # AdamW
    lr:float=3e-4
    betas:tuple[float,float]=(0.9,0.95)
    weight_decay:float=0.01

    # train
    max_iters:int=10000
    max_l2_norm:float=1.0
    batch_size:int=128
    save_interval:int=500
    eval_interval:int=250
    eval_iters:int=16
    log_interval:int=50

    # lr_schedule
    max_learning_rate:float=lr
    min_learning_rate:float=lr/10.0
    warmup_iters:int=500
    cosine_cycle_iters:int=max_iters

    # generate
    temperature:float=0.8

    # 消融实验
    norm_type:str="pre_norm"
    use_rope:bool=True
    ffn_type:str="swiglu"
    

    
    