import torch
import torch.nn as nn
def save_checkpoint(model:nn.Module,optimizer:torch.optim.Optimizer,iteration,out):
    checkpoint={
        "model_state_dict":getattr(model, '_orig_mod', model).state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "iteration":iteration
    }

    torch.save(checkpoint,out)

def load_checkpoint(src,model:nn.Module,optimizer:torch.optim.Optimizer):
    checkpoint=torch.load(src)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])

    return checkpoint['iteration']
