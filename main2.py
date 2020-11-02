import time
import math
import scipy.io as sio                     # import scipy.io for .mat file I/O
import numpy as np                         # import numpy
import matplotlib.pyplot as plt            # import matplotlib.pyplot for figure plotting
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

K = 10
Mi=5
I=1
num_H = 25000
training_epochs = 100
trainseed = 0
testseed = 7
seed=7
Pmax=1
np.random.seed(seed)
Pini = Pmax*np.ones(K)
var_noise = 1
X=np.zeros([num_H,K*Mi])
begin_time=time.time()
for loop in range(num_H):
    CH = 1/np.sqrt(2)*(np.random.randn(K,Mi)+1j*np.random.randn(K,Mi))
    H=abs(CH)
    HH=np.reshape(H, (1,Mi*K))
    X[loop,:] = HH
sio.savemat("channle state.mat",{'X':X})
X=np.reshape(X,(num_H,K,Mi))
end_time=time.time()
print(end_time-begin_time)



p_opt_num_H = np.zeros([num_H, K])
opt_process=np.zeros([num_H,200])
begin_time=time.time()
for loop in range(num_H):
    print(loop)
    vnew=0
    z=np.sqrt(Pini)
    y=np.zeros([K,Mi])
    x=np.zeros(K)
    p_opt_once=np.zeros(K)
    for i in range(K):
        y[i,:]=np.transpose(z)@X[loop,]/(z@X[loop,]@np.transpose(X[loop,])@np.transpose(z)+np.square(var_noise))
        #print(y.shape)=(10, 5)
        x[i]=1/(1-y[i,:]@np.transpose(X[loop,i,])*z[i])
        #print(x.shape)=(10,)
        vnew=vnew+math.log2(x[i])
    #print(vnew)
    VV=np.zeros(200)
    for iter in range(200):
        vold=vnew
        #print(vold)
        for i in range(K):
            btmp = x[i]*y[i,]@X[loop,i,]/sum(x@y@np.transpose(X[loop,])@X[loop,]@np.transpose(y))
            #print(btmp.shape)
            z[i] = min(btmp, np.sqrt(Pmax)) + max(btmp, 0) - btmp

        vnew = 0
        for i in range(K):
            y[i,:]=np.transpose(z)@X[loop,]/(z@X[loop,]@np.transpose(X[loop,])@np.transpose(z)+np.square(var_noise))
            x[i]=1/(1-y[i,:]@np.transpose(X[loop,i,])*z[i])
            vnew = vnew + math.log2(x[i])
        VV[iter] = abs(vnew - vold)
        if abs(vnew - vold)<= 1e-3:
            break
    #print(VV)
    opt_process[loop,:]=VV
    p_opt_once=np.square(z)
    #print(p_opt_once.shape)=(10,)
    p_opt_num_H[loop,:]=p_opt_once
end_time=time.time()
print(opt_process,opt_process.shape)  # (25000,200)
sio.savemat("objective function.mat",{'opt_process':opt_process})
print(p_opt_num_H,p_opt_num_H.shape)#(25000, 10)
sio.savemat("optimal power.mat",{'p_opt_num_H':p_opt_num_H})
print(end_time-begin_time)
