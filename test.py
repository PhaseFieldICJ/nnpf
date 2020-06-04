"""importation des modules """
# matplotlib 
import matplotlib.pyplot as plt 

# Default interpolation method when using imshow
plt.rcParams["image.interpolation"] = 'bilinear'  
plt.rcParams["figure.figsize"]=(5,5)

# numpy 
import numpy as np
from numpy import pi
from numpy.fft import fft2, ifft2, ifftshift

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
device=th.device("cuda:0" if th.cuda.is_available() else "cpu") #CPU ou GPU
th.set_printoptions(precision=8) #précison à 1e-8

k_size=17 # Taille noyau dans le réseau

"""Domaine """
## Domaine en espace
X,Y=np.meshgrid(np.linspace(-1/2,1/2,N), np.linspace(-1/2,1/2,N))

## Domaine de Fourier 
k=[i for i in range(0,N//2+1)]+[-i for i in range(N//2-1,0,-1)] # concaténation 
k=np.array(k)
K1,K2=np.meshgrid(k,k)

# Noyau de la chaleur en domain spatial
def kernel(dt):
    k=np.exp(-4*(pi**2)*(K1**2+K2**2)*dt)
    return ifftshift(ifft2(k).real)


## Profil champ de phase 
def profil(x):
    return (1-np.tanh(x/(2*epsilon)))/2


# Distance signée cercle pour flot par courbure moyenne
def dist_MC(r,t):
    return np.sqrt(X**2+Y**2)-np.sqrt(r**2-2*t)

"""paramètre splitting convexe """
alpha=2 # >1

""" Opérateurs """
# diffusion 
def convex_diff(u, dt=dt, alpha=alpha):
    
    #Noyau du convexe splitting dans le domain de fourier
    convex_kernel=1/(1+alpha+4*pi**2*(K1**2+K2**2)*dt)
    
    return ifft2(fft2(u)*convex_kernel).real

# Partie réaction 
Wp=lambda x: x*(x-1)*(2*x-1)

def rho(u,dt=dt, alpha=alpha):
    return (1+alpha)*u-dt*Wp(u)/epsilon**2



######################################################################################################
""" Initialisation des modèles à partir de modèles déjà entraînés """

# Modèle Lie: réaction-diffusion 
MCF1=Lie() 
checkpoint1 = th.load('MCF1.pth')
MCF1.load_state_dict(checkpoint1['model'])
MCF1.eval()


## Modèle Lie: Diffusion-Réaction
MCF2=Lie1()
checkpoint2 = th.load('MCF2.pth') 
MCF2.load_state_dict(checkpoint2['model'])
MCF2.eval()


###################################################################################################
""" validation aire et/ou rayon """

# Data: validation 
r=0.35
d0=np.sqrt(X**2+Y**2)-r
u0=profil(d0)



nb_iter=1002
tps=[i*dt for i in range(nb_iter)]
aire0=np.empty((nb_iter,))
aire1=[pi*(r**2-2*t) for t in tps]
aire2=np.empty((nb_iter,))
aire3=[profil(dist_MC(r, t)).mean() for t in tps]

u=u0
v=u0

for i in range(nb_iter):
    aire0[i]=u.mean()
    aire2[i]=v.mean()
    with th.no_grad():
        u=to_array(MCF1(to_tensor(u)))
    v=convex_diff(rho(v))
    
plt.figure()
plt.title('Comparaison: aires ')
plt.plot(tps, aire0, label='masse réseau')
plt.plot(tps, aire1, label='aire exacte')
plt.plot(tps, aire2, label='masse convexe alpha={}'.format(alpha))
plt.plot(tps, aire3, label='masse exacte')
plt.legend()



###########################################################################################################
""" Animation """
# fig, ax=plt.subplots()

# u=u0

# graph=ax.imshow(u, clim=(0,1), cmap='jet', animated=True, interpolation='none')
# fig.colorbar(graph, ax=ax)

# for i in range(500):
#     if i%5==0:
#         graph.set_array(u)
#         ax.set_title('iter{}'.format(i))
#         fig.canvas.draw()
#         plt.pause(0.001)
#     with th.no_grad():
#         u=to_array(MCF5(to_tensor(u)))





