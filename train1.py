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
X=sio.loadmat("channel state.mat",{'X':X})
X=X['X']
X=np.reshape(X,(num_H,K,Mi))
print(X,X.shape)
p_opt_num_H = np.zeros([num_H, K])
p_opt_num_H=sio.loadmat("optimal power.mat",{'p_opt_num_H':p_opt_num_H})
p_opt_num_H=p_opt_num_H['p_opt_num_H']
print(p_opt_num_H,p_opt_num_H.shape)


IN=np.reshape(X[:20000,:,:],(20000,K*Mi))
IN=tf.convert_to_tensor(IN,dtype=tf.float32)
IN=IN/tf.reduce_max(IN)
#print(IN,IN.shape)
OUT=np.reshape(p_opt_num_H[:20000,:],(20000,K))
OUT=tf.convert_to_tensor(OUT,dtype=tf.float32)
OUT=OUT/tf.reduce_max(OUT)
#OUT=np.ones([20000, K])*0.01
#OUT=tf.convert_to_tensor(OUT,dtype=tf.float32)
#print(OUT,OUT.shape)

IN_test=np.reshape(X[20000:,:,:],(5000,K*Mi))
IN_test=tf.convert_to_tensor(IN_test,dtype=tf.float32)
IN_test=IN_test/tf.reduce_max(IN_test)
#print(IN_test,IN_test.shape)

OUT_test=np.reshape(p_opt_num_H[20000:,:],(5000,K))
OUT_test=tf.convert_to_tensor(OUT_test,dtype=tf.float32)
OUT_test=OUT_test/tf.reduce_max(OUT_test)
#print(OUT_test,OUT_test.shape)


train_db=tf.data.Dataset.from_tensor_slices((IN,OUT)).batch(1)
test_db=tf.data.Dataset.from_tensor_slices((IN_test,OUT_test)).batch(1)

train_iter=iter(train_db)
sample=next(train_iter)

w1=tf.Variable(tf.random.truncated_normal([K*Mi,2000],stddev=0.1))
b1=tf.Variable(tf.zeros([2000]))
w2=tf.Variable(tf.random.truncated_normal([2000,500],stddev=0.1))
b2=tf.Variable(tf.zeros([500]))
w3=tf.Variable(tf.random.truncated_normal([500,500],stddev=0.1))
b3=tf.Variable(tf.zeros([500]))
w4=tf.Variable(tf.random.truncated_normal([500,K],stddev=0.1))
b4=tf.Variable(tf.zeros([K]))
lr=1e-3
tran_process=np.zeros([num_H,10])
for epoch in range(10):
    for step,(IN,OUT) in enumerate(train_db):
        with tf.GradientTape() as tape:
            h1=IN@w1+b1
            h1=tf.nn.relu(h1)
            h2=h1@w2+b2
            h2=tf.nn.relu(h2)
            h3=h2@w3+b3
            h3=tf.nn.relu(h3)
            out=h3@w4+b4
            #print(out.shape)=(1,10)
            loss=(OUT-out)@np.transpose(OUT-out)
            #print(loss,loss.shape)=(1,1)
            loss=tf.reduce_mean(loss)
        grads=tape.gradient(loss,[w1,b1,w2,b2,w3,b3,w4,b4])
        w1.assign_sub(lr*grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])
        w4.assign_sub(lr * grads[6])
        b4.assign_sub(lr * grads[7])
        if step %5000 ==0:
            print(epoch,step,'loss:',float(loss))
        tran_process[step,epoch]=loss
print(tran_process,tran_process.shape)
sio.savemat("descending process.mat",{'tran_process':tran_process})

begin_time=time.time()
total_correct=np.zero000,K])
for step,(IN_test,OUT_test) in enumerate(test_db):
    h1=tf.nn.relu(IN_test@w1+b1)
    h2=tf.nn.relu(h1@w2+b2)
    h3=tf.nn.relu(h2@w3+b3)
    out=h3@w4+b4
    out=tf.abs(out)
    correct=(out-OUT_test)@np.transpose(out-OUT_test)
    total_correct[step,]=correct/(OUT_test@np.transpose(OUT_test))
print(total_correct,total_correct.shape)
sio.savemat("testing error.mat",{'total_correct':total_correct})
print(np.max(total_correct),np.min(total_correct),np.mean(total_correct))
end_time=time.time()
print(end_time-begin_time)


