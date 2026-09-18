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
from cs336_basics.optimizer import AdamW,lr_cosine_schedule,Muon
from cs336_basics.serialization import save_checkpoint,load_checkpoint
from cs336_basics.log_local import log_save_to_disk

def parse_args(config:Config):
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
            default=config.lr
        )

    parser.add_argument(
                "--lr_muon",
                type=float,
                default=config.lr_muon
            )

    parser.add_argument(
            "--batch_size",
            type=int,
            default=config.batch_size
        )

    parser.add_argument(
            "--max_iters",
            type=int,
            default=config.max_iters
        )

    parser.add_argument(
            "--checkpoint_dir",
            type=str,
            default="output/checkpoint/"
        )

    parser.add_argument(
        "--save_interval",
        type=int,
        default=config.save_interval
    )

    parser.add_argument(
        "--eval_interval",
        type=int,
        default=config.eval_interval
    )

    parser.add_argument(
        "--eval_iters",
        type=int,
        default=config.eval_iters
    )

    parser.add_argument(
        "--log_interval",
        type=int,
        default=config.log_interval
    )

    parser.add_argument(
        "--resume_checkpoint_path",
        type=str,
        default=None
    )

    parser.add_argument(
            "--norm_type",
            type=str,
            default=config.norm_type
        )


    parser.add_argument(
            "--use_rope",
            type=lambda x:str(x).lower() in ('true',),
            default=config.use_rope
        )

    parser.add_argument(
            "--ffn_type",
            type=str,
            default=config.ffn_type
        )

    parser.add_argument(
            "--use_wandb",
            type=lambda x:str(x).lower() in ('true',),
            default=config.use_wandb
        )

    parser.add_argument(
                "--use_local_log",
                type=lambda x:str(x).lower() in ('true',),
                default=config.use_local_log
            )

    parser.add_argument(
                    "--log_output_dir",
                    type=str,
                    default="output/log/"
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
    args=parse_args(config)
    arg_dict=vars(args)
    for arg,value in arg_dict.items():
        if hasattr(config,arg):
            setattr(config,arg,value)
        
    setattr(config,"cosine_cycle_iters",config.max_iters)
    setattr(config,"warmup_iters",config.max_iters//20)
    setattr(config, "max_learning_rate", config.lr)
    setattr(config, "min_learning_rate", config.lr / 10.0)
    setattr(config,"max_learning_rate_muon",config.lr_muon)
    setattr(config,"min_learning_rate_muon",config.lr_muon/10.0)

    train_tokenized_data=np.load(args.train_data_path,mmap_mode='r')
    valid_tokenized_data=np.load(args.valid_data_path,mmap_mode='r')

    model=Model.Transformer_LM(config).to(config.device)
    adam_params=[]
    muon_params=[]
    for name,param in model.named_parameters():
        if param.ndim!=2:
            adam_params.append(param)
            continue
        if any(keyword in name for keyword in ["ln","embedding","lm_head"]):
            adam_params.append(param)
        else:
            muon_params.append(param)

    opt_adam=AdamW(params=adam_params,lr=config.lr,betas=config.betas,weight_decay=config.weight_decay)
    opt_muon=Muon(params=muon_params,lr=config.lr_muon,beta=config.beta,weight_decay=config.weight_decay)

    model=torch.compile(model)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_output_dir, exist_ok=True) 
    exp_path=f"norm_type-{config.norm_type}_use_rope-{config.use_rope}_ffn_type-{config.ffn_type}_opt-muon_max_iters-{config.max_iters}_lr-{config.lr}_lr_muon-{config.lr_muon}_batch_size-{config.batch_size}"
    base_save_path=os.path.join(args.checkpoint_dir,exp_path)
    start_iter=0
    if args.resume_checkpoint_path is not None:
        start_iter=load_checkpoint(args.resume_checkpoint_path,model,opt_adam,opt_muon)

    if config.use_wandb:
        import wandb
        wandb.init(
            project="cs336-assignment1",        
            name=exp_path,    
            config=vars(config)                 
        )
    
    model.train()
    start_time=time.time()

    for iter in range(start_iter,config.max_iters):
        if iter%config.save_interval==0:
            save_checkpoint(model,opt_adam,iter,base_save_path+f"_iter-{iter}.pt",opt_muon)
            print(f"检查点已成功保存到: {base_save_path}_iter-{iter}.pt")

        if iter%config.eval_interval==0:
            eval_loss=estimate_loss(model,valid_tokenized_data,config.eval_iters,config)
            current_time=time.time()
            elapsed=current_time-start_time
            if config.use_wandb:
                wandb.log({
                    "eval/loss": eval_loss.item(),
                    "eval/ppl": torch.exp(eval_loss).item(),
                    "time/elapsed_sec": elapsed
                }, step=iter)

            if config.use_local_log:
                save_log={
                    "Iter":iter,
                    "Elapsed":elapsed,
                    "Eval loss":eval_loss.item(),
                    "Eval ppl":torch.exp(eval_loss).item(),
                }
                log_save_to_disk(os.path.join(args.log_output_dir,exp_path+".jsonl"),save_log)
            print(f"第{iter}轮eval_loss: {eval_loss}")
            print(f"第{iter}轮PPL: {torch.exp(eval_loss).item(): .4f}")


        opt_adam.zero_grad()
        opt_muon.zero_grad()
        inputs,targets=get_batch(train_tokenized_data,config)
        logits=model(inputs)
        loss=cross_entropy(logits,targets)
        loss.backward()
        global_grad=gradient_clipping(model.parameters(),config.max_l2_norm)
        lr = lr_cosine_schedule(iter, config.max_learning_rate, config.min_learning_rate, config.warmup_iters, config.cosine_cycle_iters) 
        lr_muon=lr_cosine_schedule(iter, config.max_learning_rate_muon, config.min_learning_rate_muon, config.warmup_iters, config.cosine_cycle_iters) 
        for param_group in opt_adam.param_groups:
            param_group['lr']=lr
        for param_group in opt_muon.param_groups:
            param_group['lr']=lr_muon
        opt_adam.step()
        opt_muon.step()

        
        if iter%config.log_interval==0:
            current_time=time.time()
            elapsed=current_time-start_time
            token_seen=(iter+1)*config.batch_size*config.max_seq_len
            if config.use_local_log:
                save_log={
                    "Iter":iter,
                    "Elapsed":elapsed,
                    "Train loss": loss.item(),
                    "Lr":lr,
                    "Lr_muon":lr_muon,
                    "Grad norm":global_grad.item(),
                    "token seen":token_seen
                }

                log_save_to_disk(os.path.join(args.log_output_dir,exp_path+".jsonl"),save_log)
            if config.use_wandb:
                wandb.log({
                "train/loss": loss.item(),
                "train/lr": lr,
                "train/lr_muon":lr_muon,
                "train/grad_norm": global_grad.item(),
                "time/elapsed_sec": elapsed,
                "token_seen": token_seen
                }, step=iter)

            print(f"Iter: {iter}")
            print(f"Elapsed: {elapsed: .2f}s")
            print(f"Train loss: {loss.item(): .4f}")
            print(f"Lr: {lr: .6e}")
            print(f"Lr_muon: {lr_muon: .6e}")
            print(f"Grad norm: {global_grad.item(): .4f}")
            print(f"token seen: {token_seen}")

    final_val_loss = estimate_loss(model, valid_tokenized_data, config.eval_iters, config)
    print(f"最终验证集 Loss: {final_val_loss.item():.4f}")
    save_checkpoint(model, opt_adam, config.max_iters, base_save_path+f"_iter-{config.max_iters}.pt",opt_muon)
    print(f"最终检查点已保存至: {base_save_path}_iter-{config.max_iters}.pt")
    if config.use_wandb:
        wandb.log(
        {
            "eval/loss": final_val_loss.item(),
        },step=config.max_iters
        )
        wandb.finish()

    if config.use_local_log:
        save_log={
            "Iter":config.max_iters,
            "final eval loss":final_val_loss.item(),
        }
        log_save_to_disk(os.path.join(args.log_output_dir,exp_path+".jsonl"),save_log)

if __name__=="__main__":
    train_model()
