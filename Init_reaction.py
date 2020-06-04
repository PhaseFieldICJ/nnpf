# -*- coding: utf-8 -*-

"""Import modules"""
# numpy 
import numpy as np

# Pytorch 
import torch as th
import torch.nn as nn 
import torch.optim as optim

""" Hyper paramètres et autres  """
N=2**8 # taille échantillon 
epsilon=2/N # paramètres d'approximation champ de phase
dt=epsilon**2 # pas de temps

dtype=th.float32 # type des tenseurs
device=None #CPU ou GPU


"""Converion """

def to_tensor(array, dtype=th.float32):
    return th.tensor(array, dtype=dtype)

"""Solution exacte partie reaction

Condition initiale: l'identité sur [0,1]
"""  

def c(s,dt):
    return np.exp(-dt/epsilon**2)*s*(1-s)/(1-2*s)**2

def sol(s,dt):
    if s==1/2:
        return 1/2
    elif s<1/2:
        a=np.sqrt(1+4*c(s,dt))
        return 1-(a+1)/(2*a)
    else:
        a=np.sqrt(1+4*c(s,dt))
        return (a+1)/(2*a)

Nb_echon=100
 
x0=np.linspace(0,1,Nb_echon)

y0=[sol(x,dt/2) for x in x0]  # pas de temps dt=epsilon^2/2
y1=[sol(x,dt) for x in x0] # pas de temps dt=epsilon^2


""" Trainer """
class trainer:
    def __init__(self, model, loss_fn, optimizer, x, y, name='checkpoint.pth'):
        self.model=model
        self.loss_fn=loss_fn
        self.optimizer=optimizer
        self.input=to_tensor(x)
        self.label=to_tensor(y)
        self.nb_iter=10
        self.name=name
        
    def run(self, epochs=1000*2, max_time=None, min_loss=1e-6):
        
        for epoch in range(epochs):
            """forward pass"""
            y_pred=self.model(self.input)
            loss=self.loss_fn(self.label, y_pred)
                
            if epoch%(self.nb_iter)==0:
                print("epoch:\t ", epoch+1 , "\t loss: \t", loss)
                    
            """Backward pass"""
            self.optimizer.zero_grad() # grad=0
            loss.backward()
            self.optimizer.step()
                
            if loss<min_loss:
                th.save(self.model.state_dict(), self.name)
                break
        th.save(self.model.state_dict(), self.name)
 
"""Réseau de réaction """


class reaction(nn.Module):
    def __init__(self, *activation):
        super().__init__()
        
        def gen_layers():
            curr_dim = 1
            for fn, dim in activation:
                yield nn.Linear(curr_dim, dim, bias=True)
                yield fn
                curr_dim = dim
            yield nn.Linear(curr_dim, 1, bias=True)
            #yield nn.Sigmoid()
            
        self.fn=nn.Sequential(*gen_layers())
        
    def forward(self, x):
        x=x.reshape(*x.shape,1)
        return self.fn(x).squeeze()
  
    
class Customfunction(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn
    def forward(self,x):
        return self.fn(x)
    
def Gaussienne():
    return Customfunction(lambda x: th.exp(-(x**2)))



react1=reaction(*[(Gaussienne(),8),(Gaussienne(),4)])
react2=reaction(*[(Gaussienne(),8),(Gaussienne(),4)])

trainer1=trainer(react1, nn.MSELoss(),
                 optim.Adam(react1.parameters(), lr=1e-3),
                 x0,y0,'reactdt.pth' )

trainer2=trainer(react2, nn.MSELoss(),
                 optim.Adam(react2.parameters(), lr=1e-3),
                 x0,y1,'reactdt2.pth' )



trainer1.run(5000*2)
trainer2.run(5000*2)



