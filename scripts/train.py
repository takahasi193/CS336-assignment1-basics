import argparse
import os
import torch
import time
import numpy as np
from cs336_basics.config import Config
import numpy.typing as npt
import cs336_basics.model as Model
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import cross_entropy,gradient_clipping
from cs336_basics.optimizer import AdamW,lr_cosine_schedule
from cs336_basics.serialization import save_checkpoint,load_checkpoint



def parse_args():
    parser=argparse.ArgumentParser(
        description="训练模型"
    )

    parser.add_argument(
        "--train_data_path",
        type=str,
        default="data/tokenized/TinyStoriesV2-GPT4-train.npy"
    )

    parser.add_argument(
            "--valid_data_path",
            type=str,
            default="data/tokenized/TinyStoriesV2-GPT4-valid.npy"
        )

    parser.add_argument(
            "--lr",
            type=float,
            default=None
        )

    parser.add_argument(
            "--batch_size",
            type=int,
            default=None
        )

    parser.add_argument(
            "--max_iters",
            type=int,
            default=None
        )

    parser.add_argument(
            "--warmup_iters",
            type=int,
            default=None
        )

    parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="output/checkpoint/"
        )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=None
    )

    parser.add_argument(
        "--eval_interval",
        type=int,
        default=None
    )

    parser.add_argument(
        "--eval_iters",
        type=int,
        default=None
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=None
    )

    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        default=None
    )

    parser.add_argument(
            "--norm_type",
            type=str,
            default=None
        )


    parser.add_argument(
            "--use_rope",
            type=lambda x:str(x).lower() in ('true',),
            default=None
        )

    parser.add_argument(
            "--ffn_type",
            type=str,
            default=None
        )
    
    return parser.parse_args()

@torch.no_grad()
def estimate_loss(model:torch.nn.Module,eval_data:npt.NDArray,eval_iters:int,config:Config):
    model.eval()
    loss=0
    for _ in range(eval_iters):
        inputs,targets=get_batch(eval_data,config)
        logits=model(inputs)
        loss+=cross_entropy(logits,targets)
    loss/=eval_iters
    model.train()
    return loss

def train_model():
    config=Config()
    args=parse_args()
    arg_dict=vars(args)
    for arg,value in arg_dict.items():
        if hasattr(config,arg) and value is not None:
            setattr(config,arg,value)
        
    setattr(config,"cosine_cycle_iters",config.max_iters)
    setattr(config, "max_learning_rate", config.lr)
    setattr(config, "min_learning_rate", config.lr / 10.0)

    train_tokenized_data=np.load(args.train_data_path,mmap_mode='r')
    valid_tokenized_data=np.load(args.valid_data_path,mmap_mode='r')

    model=Model.Transformer_LM(config).to(config.device)
    opt=AdamW(params=model.parameters(),lr=config.lr,betas=config.betas,weight_decay=config.weight_decay)
    model=torch.compile(model)

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    save_path=os.path.join(args.checkpoint_dir,r"checkpoint_iter")

    start_iter=0
    if args.resume_checkpoint_path is not None:
       start_iter=load_checkpoint(args.resume_checkpoint_path,model,opt)

    import wandb

    # 初始化项目与本次实验名称
    wandb.init(
        project="cs336-assignment1",        
        name=f"baseline_lr_{config.lr}",    
        config=vars(config)                 
    )
    
    model.train()
    start_time=time.time()
    for iter in range(start_iter,config.max_iters):
        if iter%config.save_interval==0:
            save_checkpoint(model,opt,iter,f"{save_path}_{iter}.pt")
            print(f"检查点已成功保存到: {save_path}_{iter}.pt")

        if iter%config.eval_interval==0:
            eval_loss=estimate_loss(model,valid_tokenized_data,config.eval_iters,config)
            wandb.log({
                "eval/loss": eval_loss.item(),
                "eval/ppl": torch.exp(eval_loss).item()
            }, step=iter)
            print(f"第{iter}轮loss: {eval_loss}")
            print(f"第{iter}轮PPL: {torch.exp(eval_loss).item(): .4f}")


        opt.zero_grad()
        inputs,targets=get_batch(train_tokenized_data,config)
        logits=model(inputs)
        loss=cross_entropy(logits,targets)
        loss.backward()
        global_grad=gradient_clipping(model.parameters(),config.max_l2_norm)
        lr = lr_cosine_schedule(iter, config.max_learning_rate, config.min_learning_rate, config.warmup_iters, config.cosine_cycle_iters) 
        for param_group in opt.param_groups:
            param_group['lr']=lr
        opt.step()

        
        if iter%config.log_interval==0:
            current_time=time.time()
            elapsed=current_time-start_time
            wandb.log({
            "train/loss": loss.item(),
            "train/lr": lr,
            "train/grad_norm": global_grad.item(),
            "time/elapsed_sec": elapsed
            }, step=iter)

            print(f"Iter: {iter}")
            print(f"Elapsed: {elapsed: .2f}s")
            print(f"Train loss: {loss.item(): .4f}")
            print(f"Ir: {lr: .6e}")
            print(f"Grad_norm: {global_grad.item(): .4f}")

    final_val_loss = estimate_loss(model, valid_tokenized_data, config.eval_iters, config)
    print(f"最终验证集 Loss: {final_val_loss.item():.4f}")
    save_checkpoint(model, opt, config.max_iters, f"{save_path}_final.pt")
    print(f"最终检查点已保存至: {save_path}_final.pt")
    wandb.finish()

if __name__=="__main__":
    train_model()





    
