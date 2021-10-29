import numpy as np
from numpy import ndarray

from .NeuralNetwork import NeuralNetwork
from .DenseLayer import DenseLayer
from .Sigmoid import Sigmoid
from .MeanSquaredError import MeanSquaredError
from .Linear import Linear
from .SGD import SGD
from .Trainer import Trainer

from .util import to_2d_np
from .metrics import eval_regression_model

# define the neural networks

linear_regression = NeuralNetwork(
    layers=[DenseLayer(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

neural_network = NeuralNetwork(
    layers=[DenseLayer(neurons=13,
                   activation=Sigmoid()),
            DenseLayer(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

deep_learning = NeuralNetwork(
    layers=[DenseLayer(neurons=13,
                   activation=Sigmoid()),
            DenseLayer(neurons=13,
                   activation=Sigmoid()),
            DenseLayer(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

# Read the data
from sklearn.datasets import load_boston

boston = load_boston()
data = boston.data
target = boston.target
features = boston.feature_names

# Scaling the data
from sklearn.preprocessing import StandardScaler
s = StandardScaler()
data = s.fit_transform(data)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.3, random_state=80718)

# make target 2d array
y_train, y_test = to_2d_np(y_train), to_2d_np(y_test)



### Train linear regression
print("Start training: linear regression")
trainer_linear_regression = Trainer(linear_regression, SGD(learning_rate = 0.01))

trainer_linear_regression.fit(X_train=X_train, 
                              y_train=y_train, 
                              X_test=X_test,
                              y_test=y_test,
                              epochs=100,
                              eval_every=10,
                              seed=20190501)
print()
eval_regression_model(linear_regression, X_test=X_test, y_test=y_test)


### Train neural network
print("Start training: neural network")
trainer_neural_network = Trainer(neural_network, SGD(learning_rate = 0.01))

trainer_neural_network.fit(X_train=X_train, 
                            y_train=y_train, 
                            X_test=X_test,
                            y_test=y_test,
                            epochs=100,
                            eval_every=10,
                            seed=20190501)
print()
eval_regression_model(neural_network, X_test=X_test, y_test=y_test)


### Train deep learning
print("Start training: deep learning")
trainer_deep_learning = Trainer(deep_learning, SGD(learning_rate = 0.01))

trainer_deep_learning.fit(X_train=X_train, 
                            y_train=y_train, 
                            X_test=X_test,
                            y_test=y_test,
                            epochs=1000,
                            eval_every=10,
                            seed=20190501)
print()
eval_regression_model(deep_learning, X_test=X_test, y_test=y_test)



### Train deep learning2
print("Start training: deep learning2")

deep_learning2 = NeuralNetwork(
    layers=[DenseLayer(neurons=13,
                   activation=Sigmoid()),
            # DenseLayer(neurons=13,
            #        activation=Sigmoid()),
            DenseLayer(neurons=13,
                   activation=Sigmoid()),
            DenseLayer(neurons=1,
                   activation=Linear())],
    loss=MeanSquaredError(),
    seed=20190501
)

trainer_deep_learning2 = Trainer(deep_learning2, SGD(learning_rate = 0.001))

trainer_deep_learning2.fit(X_train=X_train, 
                            y_train=y_train, 
                            X_test=X_test,
                            y_test=y_test,
                            epochs=1000,
                            eval_every=10,
                            seed=20190501)
print()
eval_regression_model(deep_learning2, X_test=X_test, y_test=y_test)

