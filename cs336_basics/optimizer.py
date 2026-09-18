import torch
from torch import Tensor
from collections.abc import Iterable
import math

class AdamW(torch.optim.Optimizer):
    def __init__(self,params:Iterable[Tensor]|list[dict],
            lr: float | Tensor = 1e-3,
            betas: tuple[float | Tensor, float | Tensor] = (0.9, 0.999),
            eps: float = 1e-8,
            weight_decay: float = 0.01):
        if lr<0.0:
            raise ValueError("learning rate can't to be a negative number")
        if betas[0]<0.0 or betas[0]>=1.0 or betas[1]<0.0 or betas[1]>=1.0:
            raise ValueError("betas need to be in the [0.0,1.0)")
        if eps<=0.0:
            raise ValueError("eps need to be a positive number")
        if weight_decay<0.0:
            raise ValueError("weight_decay can't to be a negative number")
        defaults={
            "lr":lr,
            "betas":betas,
            "eps":eps,
            "weight_decay":weight_decay
        }
        super().__init__(params,defaults)
       

    @torch.no_grad
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()

        for param_group in self.param_groups:
            lr=param_group['lr']
            betas=param_group['betas']
            eps=param_group['eps']
            weight_decay=param_group['weight_decay']
            for p in param_group['params']:
                if p.grad is None:
                    continue
                state=self.state[p]
                # 获取状态
                if len(state)==0:
                    step=state.get('step',0)
                    exp_avg=state.get('exp_avg',torch.zeros_like(p,dtype=p.dtype,device=p.device))
                    exp_avg_sq=state.get('exp_avg_sq',torch.zeros_like(p,dtype=p.dtype,device=p.device))
                else:
                    step=state['step']
                    exp_avg=state['exp_avg']
                    exp_avg_sq=state['exp_avg_sq']
                step+=1
                state['step']=step

                # 权重衰减
                p.sub_(lr*weight_decay*p)

                # 计算动量一阶矩和二阶矩
                exp_avg=betas[0]*exp_avg+(1-betas[0])*p.grad
                exp_avg_sq=betas[1]*exp_avg_sq+(1-betas[1])*p.grad*p.grad

                state['exp_avg']=exp_avg
                state['exp_avg_sq']=exp_avg_sq

                # 修正动量
                exp_avg=exp_avg/(1-betas[0]**step)
                exp_avg_sq=exp_avg_sq/(1-betas[1]**step)

                # 更新权重
                p.sub_((lr*exp_avg)/(torch.sqrt(exp_avg_sq)+eps))
        return loss

class Muon(torch.optim.Optimizer):
    def __init__(self,params:Iterable[Tensor]|list[dict],
                 lr:float|Tensor,
                 beta:float=0.95,
                 weight_decay:float=0.01,
                 K:int=5,
                 a:float=3.4445,
                 b:float=-4.7750,
                 c:float=2.0315,
                 eps:float=1e-8,
                 nesterov:bool|None=True):
        if lr<0.0:
            raise ValueError("learning rate can't to be a negative number")
        if beta<0 or beta>=1:
            raise ValueError("beta should be in range(0,1)")
        if eps<=0.0:
            raise ValueError("eps need to be a positive number")
        if weight_decay<0.0:
            raise ValueError("weight_decay can't to be a negative number")
        self.a=a
        self.b=b
        self.c=c
        self.K=K
        default={
            "lr":lr,
            "beta":beta,
            "weight_decay":weight_decay,
            "eps":eps,
        }

        self.nesterov=nesterov

        super().__init__(params,default)

    @classmethod
    @torch.compile
    def Newton_Schulz(cls,u:torch.Tensor,K:int,a:float,b:float,c:float,eps:float):
        assert u.ndim==2
        x=u/(u.norm()+eps)
        transpose=False
        if x.size(0)>x.size(1):
            transpose=True
            x=x.T
        for _ in range(K):
            A=x@x.T
            B=b*A+c*A@A
            x=a*x+B@x
        if transpose:
            x=x.T
        return x*math.sqrt(max(1,x.size(0)/x.size(1)))

    @torch.no_grad()
    def step(self,closure=None):
        loss=None
        if closure is not None:
            with torch.enable_grad():
                loss=closure()
        for param_group in self.param_groups:
            lr=param_group['lr']
            weight_decay=param_group['weight_decay']
            beta=param_group['beta']
            eps=param_group['eps']
            for p in param_group['params']:
                if p.grad is None:
                    continue
                state=self.state[p]
                if len(state)==0:
                    exp_avg=state.get('exp_avg',torch.zeros_like(p.grad))
                    step=state.get('step',0)
                else:
                    exp_avg=state['exp_avg']
                    step=state['step']
                exp_avg=beta*exp_avg+(1-beta)*p.grad
                if self.nesterov:
                    u=(1-beta)*p.grad+beta*exp_avg
                else:
                    u=exp_avg
                p.sub_(lr*weight_decay*p)
                p.sub_(lr*self.Newton_Schulz(u,self.K,self.a,self.b,self.c,eps))
                step+=1
                state['exp_avg']=exp_avg
                state['step']=step
        
        return loss
        
        


def lr_cosine_schedule(it:int,max_learning_rate,min_learning_rate,warmup_iters,cosine_cycle_iters):
    if it<=warmup_iters:
        return (it/warmup_iters)*max_learning_rate

    elif it>warmup_iters and it<=cosine_cycle_iters:
        it=it-warmup_iters
        x=(it/(cosine_cycle_iters-warmup_iters))*math.pi
        return ((max_learning_rate-min_learning_rate)*(math.cos(x)+1.0)+2*min_learning_rate)/2.0

    else:
        return min_learning_rate





