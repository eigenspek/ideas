
# Improved MNIST MLP method with a second Hidden layer with Exponential Activation 2times smaller than the previous layer
print("we had a 97.5% accurate model previously that we will go from")
print("with a new Hidden layer (explin) 2times wider we obtain a 97.1% accuracy --ABORT")
print("with a new Hidden layer (explin) 2times smaller we obtain a 96.8% accuracy --ABORT")
print("with a new Hidden layer (rectlin) 2times wider we obtain a 97.2% accuracy --ABORT")
print("with a new Hidden layer (rectlin) 2times smaller we obtain a 97.6% accuracy --BETTER")

from neon.util.argparser import NeonArgparser
parser = NeonArgparser(__doc__)
args = parser.parse_args()

from neon.data import MNIST

mnist = MNIST()
train_set = mnist.train_iter
test_set = mnist.valid_iter

from neon.initializers import Gaussian

init_norm = Gaussian(loc=0.0, scale=0.01)

from neon.layers import Affine
from neon.transforms import Rectlin, Softmax, Explin

layers = []
layers.append(Affine(nout=100, init= init_norm, activation = Rectlin()))
layers.append(Affine(nout=50, init= init_norm, activation = Rectlin()))
layers.append(Affine(nout=10, init= init_norm, activation = Softmax()))

from neon.models import Model
mlp = Model(layers=layers)

from neon.layers import GeneralizedCost
from neon.transforms import CrossEntropyMulti

cost = GeneralizedCost(costfunc = CrossEntropyMulti())

from neon.optimizers import GradientDescentMomentum

optimizer = GradientDescentMomentum(0.1, momentum_coef=0.9)

from neon.callbacks.callbacks import Callbacks

callbacks = Callbacks(mlp, eval_set=test_set, **args.callback_args)

mlp.fit(train_set, optimizer=optimizer, num_epochs=args.epochs, cost=cost, callbacks=callbacks)
from neon.transforms import Misclassification
results = mlp.get_outputs(test_set) 
error = mlp.eval(test_set, metric=Misclassification())*100
print('Misclassification error = %.1f%%' % error)


