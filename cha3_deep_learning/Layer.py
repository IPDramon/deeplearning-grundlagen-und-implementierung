import numpy as np
from numpy import ndarray
from .ParamOperation import ParamOperation, Operation
from .util import assert_same_shape
from typing import List

class Layer(object):
  '''
  One layer of the neural network
  '''

  def __init__(self, neurons: int) -> None:
    '''
    The number of neurons is equal to the breadth of this layer
    '''
    self.neurons = neurons
    self.first = True
    self.params: List[ndarray] = []
    self.param_grads: List[ndarray] = []
    self.operations: List[Operation] = []


  def _setup_layer(self, num_in: int) -> None:
    '''
    Must be implemented for every layer
    '''
    raise NotImplementedError()


  def forward(self, input_: ndarray) -> ndarray:
    '''
    Sends the input through all operations
    '''
    if self.first:
      self._setup_layer(input_)
      self.first = False

    self.input_ = input_

    for operation in self.operations:
      input_ = operation.forward(input_)

    self.output = input_

    return self.output


  def backward(self, output_grad: ndarray) -> ndarray:
    '''
    Calls the backpropagation for every operation
    '''
    assert_same_shape(self.output, output_grad)

    for operation in reversed(self.operations):
      output_grad = operation.backward(output_grad)

    input_grad = output_grad

    self._param_grads()

    return input_grad


  def _param_grads(self) -> ndarray:
    '''
    Extracts the param gradient from all operations in a layer
    '''
    self.param_grads = []
    for operation in self.operations:
      if issubclass(operation.__class__, ParamOperation):
        self.param_grads.append(operation.param_grad)


  def _params(self) -> ndarray:
    '''
    Extracts the paramas from all operations in a layer
    '''
    self.params = []
    for operation in self.operations:
      if issubclass(operation.__class__, ParamOperation):
        operation.params.append(operation.param)