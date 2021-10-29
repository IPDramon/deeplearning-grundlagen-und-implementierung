import numpy as np
from numpy import ndarray
from .util import assert_same_shape
from .ParamOperation import ParamOperation

class BiasAdd(ParamOperation):

  def __init__(self, B: ndarray) -> None:
    super().__init__(B)

  
  def _output(self) -> ndarray:
    return self.input_ + self.param

  def _input_grad(self, output_grad: ndarray) -> ndarray:
    return np.ones_like(self.input_) * output_grad

  def _param_grad(self, output_grad: ndarray) -> ndarray:
    param_grad = np.ones_like(self.param) * output_grad
    return np.sum(param_grad, axis=0).reshape(1, param_grad.shape[1])