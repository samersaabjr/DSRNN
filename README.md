This repository contains implementations for the dynamically stabilized recurrent neural network (DSRNN) proposed in "A Dynamically Stabilized Recurrent Neural Network" in Tensorflow.

This novel recurrent neural network (RNN) architecture includes learnable skip-connections across a specified number of time-steps, which allows for a state-space representation of the network’s hidden-state trajectory, and introduces a regularization term in the loss function by utilizing the Lyapunov stability theory. The regularizer enables placement of eigenvalues of the (linearized) transfer function matrix to desired locations in the complex plane, thereby acting as an internal controller for the hidden-state trajectories.

The DSRNN cell can be found in the file "c_k_rnn.py", which is used to design the DSRNN which is run and trained in the file "main_DSRNN.py". All supplemental functions used can be found in the file "sup_functions.py".
