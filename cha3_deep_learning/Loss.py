import numpy as np
from numpy import ndarray
from .util import assert_same_shape

class Loss(object):
  '''
  The "loss" of a neural network
  '''

  def __init__(self):
    '''Pass'''
    pass

  def forward(self, prediction: ndarray, target: ndarray) -> float:
    '''
    Computes the actual loss value
    '''
    assert_same_shape(prediction, target)

    self.prediction = prediction
    self.target = target

    loss_value = self._output()

    return loss_value

  def backward(self) -> ndarray:
    '''
    Computes gradient of the loss value with respect to the input to the loss function
    '''
    self.input_grad = self._input_grad()

    assert_same_shape(self.prediction, self.input_grad)

    return self.input_grad

  def _output(self) -> float:
    '''
    Every subclass of "Loss" must implement the _output function.
    '''
    raise NotImplementedError()

  def _input_grad(self) -> ndarray:
    '''
    Every subclass of "Loss" must implement the _input_grad function.
    '''
    raise NotImplementedError()