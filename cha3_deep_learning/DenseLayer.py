import numpy as np
from numpy import ndarray
from .Layer import Layer, Operation
from .Sigmoid import Sigmoid
from .WeightMultiply import WeightMultipy
from .BiasAdd import BiasAdd

class DenseLayer(Layer):
  '''
  A completely connected layer of neurons
  '''
  def __init__(self,
               neurons: int,
               activation: Operation = Sigmoid()) -> None:
    super().__init__(neurons)
    self.activation = activation

  
  def _setup_layer(self, input_: ndarray) -> None:
    if self.seed:
      np.random.seed(self.seed)

    self.params = []

    #weights
    self.params.append(np.random.randn(input_.shape[1], self.neurons))

    #bias
    self.params.append(np.random.randn(1, self.neurons))

    self.operations = [WeightMultipy(self.params[0]),
                       BiasAdd(self.params[1]),
                       self.activation]

    #Not needed
    return None