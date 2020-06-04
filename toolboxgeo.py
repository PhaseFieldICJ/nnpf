# -*- coding: utf-8 -*-

# numpy 
import numpy as np
from numpy import pi
from numpy.fft import fft2, ifft2, ifftshift

# Pytorch 
import torch as th
import torch.nn as nn 
import torch.optim as optim

import time 

""" Conversion """

def to_tensor(array, dtype=th.float32):
    return th.tensor(array, dtype=dtype)
def to_array(tensor):
    return tensor.detach().numpy()

""" tronquer """
def truncate(array, size):
    """
    Suppose array has symetric shape 
    truncate array from center with size =size
    
    """
    N=array.shape[0]
    return array[N//2-(size-1)//2:N//2+(size-1)//2+1,
                 N//2-(size-1)//2:N//2+(size-1)//2+1]


""" trainer """
class trainer:
    def __init__(self, model, loss_fn, optimizer, x, y, name='checkpoint.pth'):
        self.model=model
        self.loss_fn=loss_fn
        self.optimizer=optimizer
        self.input=to_tensor(x)  # size=[nb_samples, shape]
        self.label=to_tensor(y)  # size=[nb_samples, shape]
        
        self.nb_epoch=1
        self.name=name
        self.loss=th.tensor([0])
        self.epochs=0
        
    def run(self, epochs=1000, nb_des=1, max_time=None, min_loss=0):
        
        for epoch in range(epochs):
            iteration=0
            for x,y in zip(self.input,self.label):# à vérifier
                for i in range(nb_des):
                    """forward pass"""
                    y_pred=self.model(x)
                    self.loss=self.loss_fn(y, y_pred)
                    
                
                    if epoch%(self.nb_epoch)==0 and iteration%1==0:
                        print("epoch:\t ", epoch ,"\t itération:\t", 
                              iteration,"\t nb_des:\t",i, "\t loss: \t", self.loss)
                    
                    """Backward pass"""
                    self.optimizer.zero_grad() # grad=0
                    self.loss.backward()
                    self.optimizer.step()
                    
                    if self.loss<min_loss:
                        self.epochs=epoch
                        th.save({'model': self.model.state_dict(),
                                 'epoch': epoch, 
                                 'loss': self.loss
                                 }, self.name)
                        break
                iteration+=1
                
        self.epochs=epochs
        th.save({'model': self.model.state_dict(),
                             'epoch': epoch, 
                            'loss': self.loss
                            }, self.name)


#######################################################################################""""""
""" Réseaux de neurones """

class Customfunction(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn
    def forward(self,x):
        return self.fn(x)
    
def Gaussienne():
    return Customfunction(lambda x: th.exp(-(x**2)))



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

"""Réseau de diffusion """

class diffusion(nn.Module):
    def __init__(self, d_in=1, d_out=1, k_size=2**4+1, stride=1, padding=2**3,kernel=None, bias=None):
        super().__init__()
        
        self.conv=nn.Conv2d(d_in, d_out, k_size, stride=stride, padding=padding, bias=bias)
        if kernel is not None:
            N=kernel.shape[0]
            self.conv.weight=nn.Parameter(to_tensor(kernel).reshape(1,1,k_size, k_size))
        
    def forward(self, x):
        x=x.reshape(1,1,*x.shape)
        return self.conv(x).squeeze()
    
    
""" Réseau de splitting """    
    
class splitting(nn.Module):
    def __init__(self, kernel=None, react='reactdt.pth'):
        super().__init__()
        self.fn=nn.Sequential(reaction(*[(Gaussienne(),8),(Gaussienne(),4)]),
                              diffusion(kernel=kernel),
                              reaction(*[(Gaussienne(),8),(Gaussienne(),4)]))
        self.fn[0].load_state_dict(th.load(react))
        self.fn[2].load_state_dict(th.load(react))
        
    def forward(self,x):
        return self.fn(x).squeeze()

    
class Lie(nn.Module):
    def __init__(self, kernel=None, react='reactdt2.pth'):
        super().__init__()
        self.fn=nn.Sequential(reaction(*[(Gaussienne(),8),(Gaussienne(),4)]),
                              diffusion(kernel=kernel))
        self.fn[0].load_state_dict(th.load(react))
        
    def forward(self, x):
        # self.fn[1].conv.weight=nn.Parameter(th.max(self.fn[1].conv.weight,
        #                                            th.zeros_like(self.fn[1].conv.weight)))
        # self.fn[1].conv.weight=nn.Parameter(self.fn[1].conv.weight/self.fn[1].conv.weight.sum())
        return self.fn(x).squeeze()
        


class Lie1(nn.Module):
    def __init__(self, kernel=None, react='reactdt2.pth'):
        super().__init__()
        self.fn=nn.Sequential(diffusion(kernel=kernel),reaction(*[(Gaussienne(),8),(Gaussienne(),4)]))
        self.fn[1].load_state_dict(th.load(react))
    
    def forward(self, x):
        self.fn[0].conv.weight=nn.Parameter(self.fn[0].conv.weight/self.fn[0].conv.weight.sum())
        y=self.fn(x).squeeze()
        return y
    

"""Réseau global """

"""Réseau [2,1]"""
class network1(nn.Module):
    def __init__(self, kernel1=None, kernel2=None, react1='reactdt.pth', react2='reactdt2.pth'):
        super().__init__()
        # first hidden layer 
        self.net1=splitting(kernel=kernel1, react=react1)
        self.net2=splitting(kernel=kernel2, react=react2)
        self.Linear=nn.Linear(2,1)
        
        #self.Linear.weight=nn.Parameter(to_tensor([[2,-1]]))
        
        # second hidden layer 
        #self.net3=splitting()
        
    def forward(self, x):
        y1=self.net1(x)
        y2=self.net2(x)
        z=self.Linear(th.stack((y1,y2),-1))
        #return self.net3(z)
        return z.squeeze()
    
    
""" Réseau [4;2;1] """
class network2(nn.Module):
    def __init__(self, kernel1=None, kernel2=None, kernel3=None, kernel4=None,
                 kernel5=None, kernel6=None,react1='reactdt.pth', react2='reactdt.pth',
                 react3='reactdt.pth', react4='reactdt.pth', react5='reactdt.pth', react6='reactdt.pth'):
        super().__init__()
        # first hidden layer 
        self.net1=splitting(kernel=kernel1,react=react1)
        self.net2=splitting(kernel=kernel2,react=react2)
        self.net3=splitting(kernel=kernel3,react=react3)
        self.net4=splitting(kernel=kernel4,react=react4)
        self.Linear1=nn.Linear(4,2)
        
        #self.Linear.weight=nn.Parameter(to_tensor([[2,-1]]))
        
        # second hidden layer 
        self.net5=splitting(kernel=kernel5, react=react5)
        self.net6=splitting(kernel=kernel6, react=react6)
        self.Linear2=nn.Linear(2,1)
        
    def forward(self, x):
        x1=self.net1(x)
        x2=self.net2(x)
        x3=self.net3(x)
        x4=self.net4(x)
        y=self.Linear1(th.stack((x1,x2,x3,x4),-1))
        y1=self.net5(y[...,0])
        y2=self.net5(y[...,1])
        z=self.Linear2(th.stack((y1,y2),-1))
        return self.net3(z).squeeze()
    
    
"""Réseau [2,1] basé sur des briques de splitting de Lie1: Diffusion-->Réaction"""
class network3(nn.Module):
    def __init__(self, kernel1=None, kernel2=None, kernel3=None,kernel4=None,
                 react1='reactdt2.pth',react2='reactdt2.pth',
                 react3='reactdt2.pth' ,react4='reactdt2.pth'):
        super().__init__()
        
        # first layer 
        self.net1=Lie1(kernel=kernel1, react=react1)
        self.net2=Lie1(kernel=kernel2, react=react2)
        self.Linear1=nn.Linear(2, 2, bias=False)
        
        self.Linear1.weight=nn.Parameter(th.eye(2)) 
        
        # Second layer 
        self.net3=Lie1(kernel=kernel3, react=react3)
        self.net4=Lie1(kernel=kernel4, react=react4)
        self.Linear2=nn.Linear(2, 1, bias=False) 
        self.Linear2.weight=nn.Parameter(to_tensor([[2,-1]]))
        
    def forward(self, x):
        x1=self.net1(x)
        x2=self.net2(x)
        y=self.Linear1(th.stack((x1,x2),-1))
        y1=self.net3(y[...,0])
        y2=self.net4(y[...,1])
        z=self.Linear1(th.stack((y1,y2),-1))
        return z.squeeze()

""" Réseau [2,1] pour Allen-Cahn""" 
class network4(nn.Module):
    def __init__(self, kernel1=None, kernel2=None, 
                 react1='reactdt2.pth', react2='reactdt2.pth'):
        super().__init__()
        # First hidden layer 
        self.net1=Lie1(kernel=kernel1, react=react1)
        self.net2=Lie1(kernel=kernel2, react=react2)
        self.Linear=nn.Linear(2, 1, bias=True)
        
    def forward(self, x):
        x1=self.net1(x)
        x2=self.net2(x)
        y=self.Linear(th.stack((x1,x2),-1))
        return y.squeeze()