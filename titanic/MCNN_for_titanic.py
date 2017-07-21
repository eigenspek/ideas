#! coding: utf-8
import os
import sys
sys.path.append('/home/c3k4/.local/lib/python2.7/site-packages') # just because Pandas was not installed in Neon Env 
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Acquiring Pre-processed data made with preproc_data_for_MLP.py

train_set = pd.read_csv('~/Desktop/Programmes/python/titanic/titanic_data/preproc_train.csv')
goal_set = pd.read_csv('~/Desktop/Programmes/python/titanic/titanic_data/preproc_test.csv')

# print(train_set) # OK

# In train_set, we need to split features(X) from labels(y, here "Survived")

features = train_set.drop(['Survived'], axis=1)
labels = train_set['Survived']
# now we need to split the 891 cases into 800 for training and 91 for testing
Train_features = features[0:800]
Train_labels = labels[0:800]
Test_features = features[801:]
Test_labels = labels[801:]

# just so the code is clear, we rename goal_set : 
Goal_features = goal_set

# Now we need to transform these DataFrames into Numpy Arrays : using Pandas' as_matrix()
# see documentation at : https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.as_matrix.html

train_features_np = Train_features.as_matrix(columns=None)
#print(train_features_np.shape) # OK ! 
train_labels_np = Train_labels.as_matrix(columns=None)
# print(train_labels_np.shape) # OK ! 
test_features_np = Test_features.as_matrix(columns=None)
test_labels_np = Test_labels.as_matrix(columns=None)

# Now we can start to build our Neon Code : 
# generate backend : 
from neon.backends import gen_backend
be = gen_backend(backend='cpu', batch_size = 32)
print be

# Now we build the Iterator : 
from neon.data import ArrayIterator 
#nclass is the number of categories (when speaking in terms of image classification) so check 2 for dead/alive or 1 for 'status'
TRAIN = ArrayIterator(X = train_features_np, y=train_labels_np, nclass=2, make_onehot=True, lshape=(10,1))
# by doing nclass = 2 and make_onehot = True, the Survived categorie would have the onehot-vector representation
# where alive = [1,0] and dead =[0,1], in this way, we may be able to obtain results such as probability = [0.8 0.2] which
# would mean that this passenger is mostlikely alive. 
# If I use nclass=1 and make_onehot= False, the network would endup on a 1 node layer which could only give 0 or 1 as an answer. 
# which I believe for now, would make us lost information. 
print(TRAIN.shape) # returns 10 
# print(TRAIN) # OK : out = <neon.data.dataiterator.ArrayIterator object at 0x7fdf66a7b810>
#same for the TEST set:
TEST = ArrayIterator(X = test_features_np, y=test_labels_np, nclass=2, make_onehot=True, lshape=(10,1))
print(TEST.shape)

# We create the initializer 
from neon.initializers import Gaussian, Uniform, Constant
init_norm = Gaussian(loc=0.0, scale=0.01)
# we build layers 

##### MODEL N°1 : the simple model
#from neon.layers import Affine
#from neon.transforms import Rectlin, Softmax
#layers = []
#layers.append(Affine(nout=10, init=init_norm, activation=Rectlin()))
#layers.append(Affine(nout=2, init=init_norm, activation=Softmax()))

####### MODEL 2 : Conv test:

#from neon.layers import Affine, Sequential, MergeMultistream, Conv, Pooling, Dropout 
#from neon.transforms import Rectlin, Softmax

#init_uni = Uniform(low=-0.1, high=0.1) 
#init_cst = Constant(0.1) #to avoid dead neurons

#layers =[]
#layers.append(Conv((2,1,10), init=init_uni, bias=init_cst, padding=0, activation=Rectlin()))
#layers.append(Affine(nout=2, init=init_norm, activation=Softmax()))


##### MODEL N°3 : 

from neon.layers import Affine, Sequential, MergeMultistream, Conv, Pooling, Dropout 
from neon.transforms import Rectlin, Softmax

init_uni = Uniform(low=-0.1, high=0.1) 
init_cst = Constant(0.1) #to avoid dead neurons 

path1 = Sequential(layers=[Conv((1,1,10), init=init_uni, bias=init_cst, padding=0, activation=Rectlin())])

path2 = Sequential(layers=[Conv((2,1,10), init=init_uni, bias=init_cst, padding=0, activation=Rectlin())])

path3 = Sequential(layers=[Conv((3,1,10), init=init_uni, bias=init_cst, padding=0, activation=Rectlin())])

layers = [MergeMultistream(layers=[path1, path2, path3], merge="stack")]
#layers.append(Affine(nout=27, init=init_norm, activation=Rectlin()))
#layers.append(Dropout(keep=0.5))
layers.append(Affine(nout=2, init=init_norm, activation=Softmax()))


# We Build the model : 
from neon.models import Model
mlp = Model(layers=layers)

# we will use the Cross Entropy as cost and SGD as optimizer : 
from neon.layers import GeneralizedCost, Multicost
from neon.transforms import CrossEntropyMulti

cost = GeneralizedCost(costfunc = CrossEntropyMulti())

from neon.optimizers import GradientDescentMomentum, Adam

optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9) # for Model 1
#optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999) # for Model 2 

from neon.callbacks.callbacks import Callbacks

callbacks = Callbacks(mlp, TRAIN)
# Training : 
mlp.fit(dataset=TRAIN, cost=cost, optimizer=optimizer,  num_epochs=1, callbacks=callbacks)

from neon.transforms import Misclassification
results = mlp.get_outputs(TEST) 
error = mlp.eval(TEST, metric=Misclassification())*100
accuracy_fp = 100 - error
print 'Model Accuracy : %.1f%%' % accuracy_fp 


