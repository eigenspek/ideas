# coding: utf-8

# NEON

# CNN for MNIST Dataset with ADAM Optimizer
# NOTE : 97.6% Accuracy with only 1 Epoch. 


# generate backend : 
from neon.backends import gen_backend
be = gen_backend(backend='cpu', batch_size = 128) #128 à la base
print be

# Importing MNIST data + Array Iterators : 
from neon.data import MNIST
mnist = MNIST()
train_set = mnist.train_iter
test_set = mnist.valid_iter

# Building Layers : 
# NOTE : Shortcuts : Conv == Linear Convolution + Bias + Activation
# 		     Affine == Linear + Bias + Activation 

# Model : Conv(Stride 1, 0-Padding) -> MaxPool(2x2) -> Conv(s1,0-p) -> MaxP(2x2) -> Densely Connected Layer (1024) -> 1-hot vector

from neon.layers import Conv, Affine, Pooling, Dropout
from neon.initializers import Uniform, Constant, Gaussian
from neon.transforms.activation import Rectlin, Softmax

init_uni = Uniform(low=-0.1, high=0.1) # try Gaussian(loc=0, scale=0.1)
init_cst = Constant(0.1) #to avoid dead neurons 
layers = []
layers.append(Conv(fshape=(5,5,32), init=init_uni, bias=init_cst, padding=4, activation=Rectlin())) #check padding and bias
layers.append(Pooling(fshape=2, strides=2)) # op : default to max in {'max','avg'} 
layers.append(Conv(fshape=(5,5,64), init=init_uni, bias=init_cst, padding=4, activation=Rectlin()))
layers.append(Pooling(fshape=2, strides=2))
layers.append(Affine(nout=1024, init=init_uni , activation=Rectlin())) #check if we need to vectorize anything
layers.append(Dropout(keep=0.5)) # Try ranging keep beetween 0.5 and 1 --> see Paper on Dropout in Neon Folder 
layers.append(Affine(nout=10, init=init_uni, activation=Softmax())) #check if softmax is OK

# Now we build the model : 
from neon.models import Model
model = Model(layers)

#cost function :
from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti
cost = GeneralizedCost(costfunc=CrossEntropyMulti())

# optimizer : we use the ADAM optimizer here 
from neon.optimizers import Adam
optimizer = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999)

#Callback : 
from neon.callbacks.callbacks import Callbacks
callbacks = Callbacks(model, train_set)

# Training : 
model.fit(dataset=train_set, cost=cost, optimizer=optimizer,  num_epochs=1, callbacks=callbacks)

# Evaluate the model performance : 
from neon.transforms import Misclassification
error_pct = 100 * model.eval(test_set, metric=Misclassification())
accuracy_fp = 100 - error_pct
print 'Model Accuracy : %.1f%%' % accuracy_fp 
