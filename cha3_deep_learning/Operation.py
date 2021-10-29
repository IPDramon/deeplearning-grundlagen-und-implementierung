import numpy as np
from numpy import ndarray
from .util import assert_same_shape

class Operation(object):
  '''
  Basic class for every operation
  '''

  def __init__(self) -> None:
    pass


  def forward(self, input_: ndarray) -> ndarray:
    '''
    Saves the input and gets the output
    '''
    self.input_ = input_

    self.output = self._output()

    return self.output


  def backward(self, output_grad: ndarray) -> ndarray:
    '''
    Calls the calcuation of input gradient and returns it
    '''
    assert_same_shape(self.output, output_grad)

    self.input_grad = self._input_grad(output_grad)

    assert_same_shape(self.input_, self.input_grad)

    return self.input_grad


  def _output(self) -> ndarray:
    '''
    To be implemented by the respective subclass
    '''
    raise NotImplementedError()


  def _input_grad(self, output_grad: ndarray) -> ndarray:
    '''
    To be implemented by the respective sebclasses
    '''
    raise NotImplementedError()