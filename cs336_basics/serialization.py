import torch
import torch.nn as nn
def save_checkpoint(model:nn.Module,optimizer:torch.optim.Optimizer,iteration,out,optimizer_second:torch.optim.Optimizer|None=None):
    checkpoint={
        "model_state_dict":getattr(model, '_orig_mod', model).state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "iteration":iteration
    }

    if optimizer_second is not None:
         checkpoint['optimizer_second_state_dict']=optimizer_second.state_dict()       

    torch.save(checkpoint,out)

def load_checkpoint(src,model:nn.Module,optimizer:torch.optim.Optimizer,optimizer_second:torch.optim.Optimizer|None=None):
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    if optimizer_second is not None:
        assert "optimizer_second_state_dict" in list(checkpoint.keys())
        optimizer_second.load_state_dict(checkpoint['optimizer_second_state_dict'])

    return checkpoint['iteration']
