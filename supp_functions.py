import tensorflow as tf
from dsrnn import DSRNNCell
import numpy as np
import os

def get_batches(X, Y, n_steps, batch_size):
    '''
    slice the mini-batches
    
    X: X_input, to be sliced
    n_steps: num of steps (in time)
    n_inputs: input features
    n_classes: output classes
    '''
    n_batches = int(len(X) / batch_size)
    # keep only integer batches
    X = X[:int(batch_size * n_batches)]
    Y = Y[:int(batch_size * n_batches)]
    # reshape
    for n in range(0, X.shape[0], batch_size):
        # inputs
        x = X[n:n+batch_size,:,:]
        # targets
        y = Y[n:n+batch_size,:]
        yield x,y
        

def load_data(dataset, forecast_steps, dir_data, seq_len):
    if dataset == 'DP':
        os.chdir(dir_data+str(seq_len))
        X_train = np.load('train_data_'+str(forecast_steps)+'.npy')
        y_train = np.load('train_labels_'+str(forecast_steps)+'.npy')
        X_test = np.load('test_data_'+str(forecast_steps)+'.npy')
        y_test = np.load('test_labels_'+str(forecast_steps)+'.npy')
        n_input = 6         #data feature diemnsion
        n_steps = X_train.shape[1]       #timesteps used in RNN
        num_classes = y_train.shape[1]
    elif dataset == 'MNIST':
        mnist = tf.keras.datasets.mnist
        (X_train, y_train),(X_test, y_test) = mnist.load_data()
        X_train, X_test = X_train / 255.0, X_test / 255.0
        y_train = np.eye(10)[y_train]
        y_test = np.eye(10)[y_test]
        n_input = 28         #data feature diemnsion
        n_steps = 28       #timesteps used in RNN
        num_classes = y_train.shape[1]
    else:
        print('Choose existing dataset')
    
    return X_train, X_test, y_train, y_test, n_input, n_steps, num_classes
       
def build_inputs(num_steps, num_classes, n_input):
    '''
    building the input layer
    
    num_steps: number of time steps in each sequence (2nd dimension)
    '''
    inputs = tf.placeholder(tf.float32, shape=(None, num_steps, n_input), name='inputs')
    targets = tf.placeholder(tf.float32, shape=(None, num_classes), name='targets')

    # add the keep_prob
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    
    return inputs, targets, keep_prob


def build_rnn(rnn_size, num_layers, batch_size, num_steps, keep_prob, c_k):
    ''' 
    building the rnn layer
        
    keep_prob: dropout keep probability
    rnn_size: number of hidden units in rnn layer
    num_layers: number of rnn layers
    batch_size: batch_size

    '''
    # build an rnn unit
    cell = DSRNNCell(rnn_size, c_k)
#    cell = forget_cell(rnn_size, c_k)
    
    # adding dropout
#    cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=keep_prob)
    
#    rnn2 = DSRNNCell(rnn_size)
#    drop2 = tf.contrib.rnn.DropoutWrapper(rnn2, output_keep_prob=keep_prob)
#    stack_rnn = [drop]
#    for _ in range(num_layers-1):
#        stack_rnn.append(drop2)
#    # stack (changed in TF 1.2)   
#    cell = tf.contrib.rnn.MultiRNNCell(stack_rnn, state_is_tuple = True)
     
    initial_state = cell.zero_state(batch_size, tf.float32)
#    initial_state = cell.zero_state(int(batch_size/num_steps), tf.float32)
    #only used the fist layer of RNN
    
    return cell, initial_state


def build_output(rnn_output , in_size, out_size, dataset, c_k):
    ''' 
    building the output layer
        
    rnn_output: the output of the rnn layer
    in_size: rnn layer reshaped size
    out_size: softmax layer size
    
    '''  
#    rnn_output = rnn_output[:,in_size*(c_k-1):]
    rnn_output = rnn_output[:, :in_size]

    with tf.variable_scope('softmax'):
        softmax_w = tf.Variable(tf.truncated_normal([in_size, out_size], stddev=0.1))
        softmax_b = tf.Variable(tf.zeros(out_size))
    # compute logits
    logits = tf.matmul(rnn_output, softmax_w) + softmax_b
    
    out = logits
    
    
    return out, logits


def build_loss(logits, targets, rnn_size, num_classes, coeff_a, c_k):
    '''
    compute loss using logits and targets
    
    logits: fully connected layer output（before softmax）
    targets: targets
    rnn_size: rnn_size
    num_classes: class size
        
    '''
#    # One-hot coding
#    y_one_hot = tf.one_hot(targets, num_classes)
#    y_reshaped = tf.reshape(y_one_hot, logits.get_shape())

#    regularizer = tf.constant(0, dtype=tf.float32, name="beta_reg")
    # Regularizer for coeffs to give desired eigenvalues
    eig_desired = tf.constant(.01, shape=[c_k*rnn_size], dtype=tf.float32)
    eig = tf.linalg.eigvalsh(coeff_a)
    beta_reg = tf.constant(1, name="beta_reg", dtype=tf.float32)
#    beta_reg = tf.get_variable("beta_reg", shape=[], initializer=tf.constant_initializer(value=1), trainable=True, dtype=tf.float32)
    regularizer = beta_reg * tf.norm(eig-eig_desired, ord='euclidean')
    
    # Softmax cross entropy loss
#    loss = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=targets)
#    loss = tf.reduce_mean(loss)
    
    targets = tf.reshape(targets, [-1, num_classes])
    loss = tf.nn.l2_loss(logits-targets, name='l2_loss')
#    loss = tf.reduce_mean(loss)
    return loss, regularizer


def build_optimizer(loss, learning_rate, grad_clip):
    
    tvars = tf.trainable_variables()
    grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), grad_clip)
    train_op = tf.train.AdamOptimizer(learning_rate)
#    train_op = tf.train.MomentumOptimizer(learning_rate, momentum=0.01)
    optimizer = train_op.apply_gradients(zip(grads, tvars))
    
    return optimizer