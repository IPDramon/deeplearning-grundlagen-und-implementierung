import numpy as np
from numpy import ndarray
from .Operation import Operation

class Sigmoid(Operation):

  def __init__(self) -> None:
    super().__init__()

  def __sigmoid(self, x: ndarray) -> ndarray:
    return 1.0 / (1.0 + np.exp(-1.0 * x))

  def _output(self) -> ndarray:
    '''
    To be implemented by the respective subclass
    '''
    return self.__sigmoid(self.input_)


  def _input_grad(self, output_grad: ndarray) -> ndarray:
    '''
    To be implemented by the respective sebclasses
    '''
    sigmoid_backward = self.output * (1.0 - self.output)
    input_grad = sigmoid_backward * output_grad
    return input_grad

