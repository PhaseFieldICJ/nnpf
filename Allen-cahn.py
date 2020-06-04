# -*- coding: utf-8 -*-
# Moodule importation 

# matplotlib 
import matplotlib.pyplot as plt 

# Default methods when using imshow
#plt.rcParams["image.interpolation"] = 'bilinear'  
plt.rcParams["figure.figsize"]=(5,5)

# numpy 
import numpy as np
from numpy import pi
from numpy.fft import fft2, ifft2, ifftshift, fftshift
import scipy.io as sio

# Pytorch 
import torch as th
import torch.nn as nn 
import torch.optim as optim

import time 

from toolboxgeo import *

"""Hyper paramètres et autres """
## Hyper-paramètres et autres

N=2**8 # taille échantillon 
epsilon=2/N # paramètres d'approximation champ de phase
dt=epsilon**2 # pas de temps

dtype=th.float32 # type des tenseurs
device=None #CPU ou GPU
th.set_printoptions(precision=8) #précison à 1e-8

k_size=17 # Taille noyau dans le réseau

"""Domaine """
## Domaine en espace
X,Y=np.meshgrid(np.linspace(-1/2,1/2,N), np.linspace(-1/2,1/2,N))

## Domaine de Fourier 
k=[i for i in range(0,N//2+1)]+[-i for i in range(N//2-1,0,-1)] # concaténation 
k=np.array(k)
K1,K2=np.meshgrid(k,k)


"""Données d'entraînement"""

## Fonction distance pour un cercle évoluant par MCF ou Willmore
def dist_MC(r,t):
    return np.sqrt(X**2+Y**2)-np.sqrt(r**2-2*t)

## Profil champ de phase 
def profil(x):
    return (1-np.tanh(x/(2*epsilon)))/2

def data_MC(r_init=0.05,r_final=0.45, dt=dt, nb_samples=10):
    # valeurs initiales 
    #nb_data=int(np.floor(r_init**2/(2*dt)))
    x_train, y_train =np.empty((nb_samples,N,N)), np.empty((nb_samples,N,N))
    
    for i in range(nb_samples):
        x_train[i]=profil(dist_MC(r_init+i*(r_final-r_init)/nb_samples,0))
        y_train[i]=profil(dist_MC(r_init+i*(r_final-r_init)/nb_samples,dt))
        
    return x_train,y_train

# Noyau de la chaleur en domain spatial
def kernel(dt):
    k=np.exp(-4*(pi**2)*(K1**2+K2**2)*dt)
    # k=1/(1+4*(pi**2)*(K1**2+K2**2)*dt)
    #return fftshift(k).real   
    return ifftshift(ifft2(k).real) #coupe les hautes fréquences 

def dirac(size):
    if size%2==0:
        print('size has to be odd')
        return
    k=np.zeros((size,size))
    k[size//2,size//2]=1
    return k


""" Data """
"""cercles """
xtrain,ytrain = data_MC(r_init=0.05, r_final=0.45,nb_samples=100) 





#######################################################################################################




""" Modèle simple basé sur un splitting de Lie """

""" Réaction diffusion """

# Initialisation du modèle
init_kernel1=truncate(kernel(dt), size=17)

MCF1=Lie(kernel=init_kernel1)


# Entraînement 
trainer1=trainer(MCF1, nn.MSELoss(), optim.Adam(MCF1.parameters(), lr=1e-3), xtrain, ytrain, name='MCF1.pth')

trainer1.run(nb_des=1,epochs=1000) 

# th.save({'model': MCF1.state_dict()},'MCF1.pth')

MCF1.eval()

"""save parameter in file.mat"""

trained_kernel=to_array(MCF1.fn[1].conv.weight.squeeze())

data_react=to_array(MCF1.fn[0](th.linspace(0,1,100)))


# Sauvegarder pour Matlab
sio.savemat('DataMCF1.mat',{'model':'Lie' ,'epochs':trainer1.epochs ,
                            'loss':trainer1.loss.item() ,  'init_kernel': truncate(init_kernel1, k_size),
                            'trained_kernel':trained_kernel,
                            'init_react':data_react})

############################################################################################################

""" Diffusion réaction """

# init_kernel2=truncate(kernel(dt), size=17)

# MCF2=Lie1(kernel=init_kernel2)

# trainer2=trainer(MCF2, nn.MSELoss(), optim.Adam(MCF2.parameters(), lr=1e-3), xtrain, ytrain, name='MCF2.pth')

# trainer2.run(epochs=30)

# th.save({'model': MCF2.state_dict()},'MCF2.pth' )

# print('last loss:', trainer2.loss)

# MCF2.eval()

# """save parameter in file.mat"""

# trained_kernel=to_array(MCF2.fn[0].conv.weight.squeeze())

# data_react=to_array(MCF2.fn[1](th.linspace(0,1,100)))

# sio.savemat('DataMCF2.mat',{'model':'Lie1' ,'epochs':trainer2.epochs ,
#                             'loss':trainer2.loss.item() ,  'init_kernel': truncate(init_kernel2, k_size),
#                             'trained_kernel':trained_kernel,
#                             'react':data_react} )

