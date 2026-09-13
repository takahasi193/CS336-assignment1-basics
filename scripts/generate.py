import torch
import torch.nn.functional as F
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





