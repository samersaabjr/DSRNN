# import tensorflow as tf
import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
import numpy as np
from supp_functions import build_inputs, build_rnn, build_output, build_loss, build_optimizer

class DSRNN:
    
    def __init__(self, num_classes, batch_size, num_steps, num_inputs,
                       rnn_size, num_layers, learning_rate, dataset_name,
                       c_k, grad_clip=5, sampling=False):
    
        # if sampling is True，use SGD, only 1 sample
        if sampling == True:
            batch_size = num_steps * 1
        else:
            batch_size, num_steps = batch_size, num_steps

        tf.reset_default_graph()
        
        # input layer
        self.inputs, self.targets, self.keep_prob = build_inputs(num_steps, num_classes, num_inputs)

        # rnn layerfrom f_gate_cell import forget_cell

        cell, self.initial_state = build_rnn(rnn_size, num_layers, batch_size, num_steps, self.keep_prob, c_k)
#        cell1, self.initial_state1 = build_rnn(rnn_size, num_layers, batch_size, num_steps, self.keep_prob, c_k)
#        cell2, self.initial_state2 = build_rnn(rnn_size, num_layers, batch_size, num_steps, self.keep_prob, c_k)
        
#        cell = tf.nn.rnn_cell.MultiRNNCell([cell1, cell2])
#        # one-hot coding for inputs
#        x_one_hot = tf.one_hot(self.inputs, num_classes)
        
        # running the RNN
        outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, initial_state=self.initial_state)
#        outputs, state = tf.nn.dynamic_rnn(cell, self.inputs, dtype=tf.float32)
        self.final_state = state
        
        # predicting the results
        self.prediction, self.logits = build_output(state, rnn_size, num_classes, dataset_name, c_k)

#        self.coeff_a = tf.constant([0])
        coeff_a_kernel = tf.reshape(outputs[rnn_size: , -1, : ], [c_k, rnn_size])
        coeff_a_eye = tf.eye(rnn_size, batch_shape=[c_k])
        self.coeff_a= tf.reshape(tf.transpose(tf.einsum('ij,ijk->ijk',coeff_a_kernel,coeff_a_eye), perm=[1,0,2]), [rnn_size, rnn_size*c_k])
        self.coeff_a = tf.concat([self.coeff_a[:,:rnn_size]+outputs[:rnn_size, -1, :], self.coeff_a[:,rnn_size:]], 1)
        if (c_k > 1) :
            self.coeff_a = tf.concat([self.coeff_a, 
                                     tf.convert_to_tensor(np.kron(np.eye(c_k-1, M=c_k), np.eye(rnn_size)), dtype=tf.float32)], 0)
        
        # Loss and optimizer (with gradient clipping)
        self.loss_nn, self.regularizer = build_loss(self.logits, self.targets, rnn_size, num_classes, self.coeff_a, c_k)
        self.loss = self.loss_nn + self.regularizer
        
        self.optimizer = build_optimizer(self.loss, learning_rate, grad_clip)
        
        self.accuracy, self.accuracy_op = tf.metrics.accuracy(labels=tf.argmax(self.targets,1),
                                                              predictions=tf.argmax(self.logits,1), name='accuracy')