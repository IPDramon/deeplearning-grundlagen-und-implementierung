from cha3_deep_learning.Loss import Loss
from typing import List
import numpy as np
from numpy import ndarray
from .Layer import Layer
from .Loss import Loss

from typing import List

class NeuralNetwork(object):
  '''
  The representation of a neural network
  '''

  def __init__(self,
               layers: List[Layer],
               loss: Loss,
               seed: float = 1) -> None:
    self.layers = layers
    self.loss = loss
    self.seed = seed
    if seed:
      for layer in layers:
        setattr(layer, "seed", self.seed)

  
  def forward(self, x_batch: ndarray) -> ndarray:
    '''
    Transfers the input through all layers of the network
    '''

    x_out = x_batch

    for layer in self.layers:
      x_out = layer.forward(x_out)

    return x_out


  def backward(self, loss_grad: ndarray) -> None:
    '''
    Backward propagation for all layers in the network
    '''
    loss_grad_in = loss_grad

    for layer in reversed(self.layers):
      loss_grad_in = layer.backward(loss_grad_in)

    return None


  def train_batch(self,
                  x_batch: ndarray,
                  y_batch: ndarray) -> float:
    '''
    Calculate resutls for a batch, calc loss and then do backward propagation
    '''

    predictions = self.forward(x_batch)
    loss_grad = self.loss.forward(predictions, y_batch)
    self.backward(self.loss.backward())

    return loss_grad


  def params(self):
    for layer in self.layers:
      yield from layer.params


  def param_grads(self):
    for layer in self.layers:
      yield from layer.param_grads