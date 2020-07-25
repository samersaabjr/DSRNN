#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 13 14:43:56 2018

@author: yxf118
"""

from tensorflow.python.ops import math_ops, special_math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import nn_ops
from tensorflow.python.ops import gen_array_ops
from tensorflow.python.ops import nn_impl
from tensorflow.python.layers import base as base_layer
from tensorflow.python.ops.rnn_cell_impl import LayerRNNCell
from tensorflow.python.eager import context
from tensorflow.python.platform import tf_logging as logging
from tensorflow.python.keras import activations
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.framework import dtypes
import scipy.special
import numpy as np

_BIAS_VARIABLE_NAME = "bias"
_WEIGHTS_VARIABLE_NAME = "kernel"
_WEIGHTS_VARIABLE_NAME_A = "coefficients"


"""Abstract object representing an RNN cell.
Every `RNNCell` must have the properties below and implement `call` with
the signature `(output, next_state) = call(input, state)`.  The optional
third input argument, `scope`, is allowed for backwards compatibility
purposes; but should be left off for new subclasses.
This definition of cell differs from the definition used in the literature.
In the literature, 'cell' refers to an object with a single scalar output.
This definition refers to a horizontal array of such units.
An RNN cell, in the most abstract setting, is anything that has
a state and performs some operation that takes a matrix of inputs.
This operation results in an output matrix with `self.output_size` columns.
If `self.state_size` is an integer, this operation also results in a new
state matrix with `self.state_size` columns.  If `self.state_size` is a
(possibly nested tuple of) TensorShape object(s), then it should return a
matching structure of Tensors having shape `[batch_size].concatenate(s)`
for each `s` in `self.batch_size`.
"""

'''
TODO:
Rewrite with tf.contrib.cudnn_rnn.CudnnRNNTanh
'''


class c_k_RNNCell(LayerRNNCell):
  """The most basic RNN cell.
  Note that this cell is not optimized for performance. Please use
  `tf.contrib.cudnn_rnn.CudnnRNNTanh` for better performance on GPU.
  Args:
    num_units: int, The number of units in the RNN cell.
    c_n: int, the number k in c_k network
    activation: Nonlinearity to use.  Default: `tanh`. It could also be string
      that is within Keras activation function names.
    reuse: (optional) Python boolean describing whether to reuse variables
     in an existing scope.  If not `True`, and the existing scope already has
     the given variables, an error is raised.
    name: String, the name of the layer. Layers with the same name will
      share weights, but to avoid mistakes we require reuse=True in such
      cases.
    dtype: Default dtype of the layer (default of `None` means use the type
      of the first input). Required when `build` is called before `call`.
    **kwargs: Dict, keyword named properties for common layer attributes, like
      `trainable` etc when constructing the cell from configs of get_config().
  """

  def __init__(self,
               num_units,
               c_n=1,
               activation=None,
               reuse=None,
               name=None,
               dtype=None,
               **kwargs):
    super(c_k_RNNCell, self).__init__(
        _reuse=reuse, name=name, dtype=dtype, **kwargs)
    if context.executing_eagerly() and context.num_gpus() > 0:
      logging.warn("%s: Note that this cell is not optimized for performance. "
                   "Please use tf.contrib.cudnn_rnn.CudnnRNNTanh for better "
                   "performance on GPU.", self)

    # Inputs must be 2-dimensional.
    self.input_spec = base_layer.InputSpec(ndim=2)

    self._c_n = c_n
    if (self._c_n == 0) and (not isinstance(self._c_n,int)):
        raise ValueError("Expected integer c_k > 0")
    self._num_units = num_units
    
    if activation:
      self._activation = activations.get(activation)
    else:
      self._activation = math_ops.tanh
#        self._activation = gen_nn_ops.relu

  @property
  def state_size(self):
    return self._c_n * self._num_units

  @property
  def output_size(self):
    return self._num_units

  @tf_utils.shape_type_conversion
  def build(self, inputs_shape):
    if inputs_shape[-1] is None:
      raise ValueError("Expected inputs.shape[-1] to be known, saw shape: %s"
                       % inputs_shape)

    input_depth = inputs_shape[-1]
    self._kernel = self.add_variable(
        _WEIGHTS_VARIABLE_NAME,
        shape=[input_depth + self._num_units, self._num_units],
        initializer=init_ops.glorot_uniform_initializer(dtype=self.dtype))   
    self._kernel_A = self.add_variable(
        _WEIGHTS_VARIABLE_NAME_A,
        shape=[self._c_n, self._num_units],
        initializer=init_ops.glorot_uniform_initializer(dtype=self.dtype))
    self._bias = self.add_variable(
        _BIAS_VARIABLE_NAME,
        shape=[self._num_units],
        initializer=init_ops.zeros_initializer(dtype=self.dtype))

    self.built = True

  def call(self, inputs, state):
    """c_k_RNN basic operations
    """
    #e.g.  scipy.special.binom(3,[0,1,2,3]) = array([1., 3., 3., 1.])
#    coeff_mat = math_ops.cast(scipy.special.binom(self._c_n, np.arange(self._c_n)) *
#                              np.power(-1, np.flip(np.arange(self._c_n))), 
#                              dtype = dtypes.float32)
    #np.power(-1, np.arange(c_n) + 1) is the (-1)^n term


    #state dimension is [batch_size, c_k * num_hidden]
    #we want [batch_size, c_k, num_hidden]
    full_state = state[:, :self._num_units*(self._c_n-1)]
    #full_state records the entire c_k timestep states, now we discard the earliest state from the previous step
    state = gen_array_ops.reshape(state, [-1, self._c_n, self._num_units])
    
    
    # tanh(W[h,x]+b)
    current_state = math_ops.matmul(array_ops.concat([inputs, state[:,0,:]], 1), self._kernel)
    current_state = nn_ops.bias_add(current_state, self._bias)
    current_state = self._activation(current_state)
    
#    #W(tanh[h],x)+b
#    current_state = array_ops.concat([inputs, self._activation(state[:,0,:])], 1)
#    current_state = math_ops.matmul(current_state, self._kernel)
#    current_state = self._activation(current_state)    
    
    
    #coeff_mat shape [c_k,], state shape [batch_size, c_k, num_hidden]
    #want to contract along the c_k dimension
#    current_state += math_ops.tensordot(coeff_mat, state, axes=[[0], [1]])
    #comment above line to get the standard rnn

#    full_state = array_ops.concat([state[:,1:,:], 
#                                   array_ops.expand_dims(current_state,1)], 
#                                   axis=1)
#    full_state = gen_array_ops.reshape(full_state, 
#                                       [-1, self._c_n * self._num_units])
    
    current_state += special_math_ops.einsum('ijk,jk->ik', state, self._kernel_A)
#    current_state = special_math_ops.einsum('ijk,jk->ik', state, self._kernel_A) + special_math_ops.einsum('ij,j->ij', current_state, (1-math_ops.reduce_sum(self._kernel_A, 0)))
    #Einstein summation, state: [batch_size, c_k, num_hidden]
    #kernel_A: [c_k, num_hidden, num_hidden], result: [batch_size, num_hidden]
    full_state = array_ops.concat([current_state, full_state], axis=1)
    
    output = array_ops.concat([self._kernel[inputs.get_shape().as_list()[1]:,:], self._kernel_A], axis=0)
    return output, full_state

  def get_config(self):
    config = {
        "num_units": self._num_units,
        "activation": activations.serialize(self._activation),
        "reuse": self._reuse,
    }
    base_config = super(c_k_RNNCell, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))