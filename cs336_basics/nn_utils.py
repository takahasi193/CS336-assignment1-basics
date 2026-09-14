from torch import Tensor
import torch
from jaxtyping import Bool, Float, Int
from collections.abc import Iterable

def cross_entropy(inputs: Float[Tensor," batch_size vocab_size"],targets: Int[Tensor, " batch_size"]):
    targets=targets.unsqueeze(-1)
    # shape (batch_size,1)
    max_value,_=torch.max(inputs,dim=-1,keepdim=True)
    shift_inputs=inputs-max_value
    logits=torch.gather(shift_inputs,dim=-1,index=targets)
    exp_inputs=torch.exp(shift_inputs)
    # shape (batch_size,1)
    sum_exp_inputs=torch.log(torch.sum(exp_inputs,dim=-1,keepdim=True))
    loss=-logits+sum_exp_inputs
    return loss.mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float):
    combined_gradient=0
    for parameter in parameters:
        if parameter.grad is not None:
            combined_gradient+=(parameter.grad*parameter.grad).sum()
    combined_gradient=combined_gradient**0.5
    if combined_gradient>max_l2_norm:
       scaled=((combined_gradient+1e-6)**-1)*max_l2_norm
       for parameter in parameters:
           if parameter.grad is not None:
            parameter.grad=parameter.grad.mul(scaled)
    return combined_gradient

            
    