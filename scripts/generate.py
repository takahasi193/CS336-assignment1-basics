import torch
import torch.nn.functional as F
import argparse
import cs336_basics.model as Model
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.config import Config
import regex
def top_p(logits:torch.Tensor,top_p:float,temperature:float):
    if temperature>0.0:
        # shape (batch_size,vocab_size)
        temp_logits=logits/temperature
    else:
        raise ValueError('temperature need to be a positive number')
    sorted_logits,logits_indices=torch.sort(temp_logits,dim=-1,descending=True)
    sorted_prob=F.softmax(sorted_logits,dim=-1)
    cumsum_prob=torch.cumsum(sorted_prob,dim=-1,dtype=torch.float32)
    mask=cumsum_prob>top_p
    mask[...,1:]=mask[...,:-1].clone()
    mask[...,0]=False
    sorted_logits=sorted_logits.masked_fill(mask==True,float('-inf'))
    top_p_prob=F.softmax(sorted_logits,dim=-1)
    result_indices=torch.multinomial(top_p_prob,num_samples=1)
    indices=torch.gather(logits_indices,dim=-1,index=result_indices)
    return indices

def parse_args(config):
    parser=argparse.ArgumentParser(
        description="生成文本"
    )

    parser.add_argument(
        "--prompt",
        type=str,
        default="There is a"
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256
    )

    parser.add_argument(
            "--temperature",
            type=float,
            default=0.8
        )

    parser.add_argument(
            "--top_p",
            type=float,
            default=config.top_p
        )

    parser.add_argument(
                    "--vocab_load_path",
                    type=str,
                    default="data/tokenizer_config_data/vocab.pkl"
                )
    
    parser.add_argument(
            "--merges_load_path",
            type=str,
            default="data/tokenizer_config_data/merges.pkl"
        )
    
    parser.add_argument(
            "--special_tokens_load_path",
            type=str,
            default="data/tokenizer_config_data/special_tokens.pkl"
        )

    parser.add_argument(
        "--checkpoint_load_path",
        type=str,
        default=f"output/checkpoint/norm_type-{config.norm_type}_use_rope-{config.use_rope}_ffn_type-{config.ffn_type}_lr-{config.lr}_batch_size-{config.batch_size}_iter-{config.max_iters}.pt"
    )

    

    return parser.parse_args()

@torch.no_grad()
def generate():
    config=Config()
    args=parse_args(config)
    pattern = r"(norm_type|use_rope|ffn_type)-([^_]+)"
    result = dict(regex.findall(pattern, args.checkpoint_load_path))
    for key,value in result.items():
        if key=="use_rope":
            value=str(value) in ('True',)
        setattr(config,key,value)
    if args.prompt=="":
        args.prompt=r"/"
    tokenizer=Tokenizer.from_files(args.vocab_load_path,args.merges_load_path,args.special_tokens_load_path)
    model=Model.Transformer_LM(config).to(config.device)
    checkpoint=torch.load(args.checkpoint_load_path,map_location=config.device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    ids=torch.tensor(tokenizer.encode(args.prompt),device=config.device).unsqueeze(0)
    for _ in range(args.max_new_tokens):
        ids_cond=ids[...,-config.max_seq_len:]
        logits=model(ids_cond)
        logits=logits[...,-1,:]
        new_token=top_p(logits,args.top_p,args.temperature)
        if new_token.item()==tokenizer.eos_token:
            break
        ids=torch.cat([ids,new_token],dim=-1)

    text=tokenizer.decode(ids[0].tolist())
    print(text)

if __name__=="__main__":
    generate()

        
    

    







