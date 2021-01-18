This repository contains implementations for the dynamically stabilized recurrent neural network (DSRNN) proposed in "A Dynamically Stabilized Recurrent Neural Network for Modeling Dynamical Systems" in Tensorflow.

This novel recurrent neural network (RNN) architecture includes learnable skip-connections across a specified number of time-steps, which allows for a state-space representation of the network’s hidden-state trajectory, and introduces a regularization term in the loss function by utilizing the Lyapunov stability theory. The regularizer enables placement of eigenvalues of the (linearized) transfer function matrix to desired locations in the complex plane, thereby acting as an internal controller for the hidden-state trajectories.

