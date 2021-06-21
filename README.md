This repository contains implementations for the dynamically stabilized recurrent neural network (DSRNN) proposed in "A Dynamically Stabilized Recurrent Neural Network" in Tensorflow.

This novel recurrent neural network (RNN) architecture includes learnable skip-connections across a specified number of time-steps, which allows for a state-space representation of the network’s hidden-state trajectory, and introduces a regularization term in the loss function by utilizing the Lyapunov stability theory. The regularizer enables placement of eigenvalues of the (linearized) transfer function matrix to desired locations in the complex plane, thereby acting as an internal controller for the hidden-state trajectories.

The dataset used to validate the DSRNN is of a recorded double pendulum experiment, which is a chaotic system that exhibits behavior of long-term unpredictability. The results show that the DSRNN outperforms both the Long Short-Term Memory (LSTM) and vanilla recurrent neural networks, and the relative mean-squared error of the LSTM is reduced by up to ~99.64%. 

![visual_of_joints_trajectories](https://user-images.githubusercontent.com/44982976/122837944-050b1480-d2c3-11eb-8909-d89ff71a298a.png)

The networks were chosen after a basic hyperparameter search, where the batch size, learning rate, and number of LSTM blocks are varied. Batch sizes of {100, 250, 500, 1000}, learning rates of {0.1, 0.01, 0.001, 0.0001}, and LSTMs with 1 and 2 stacked layers are all considered. All networks contain 128 hidden units and the weight matrices of the DSRNN models are initialized with Glorot initializers. All the networks have a batch size of 250. The DSRNN models have a learning rate of 0.0001. The LSTM and vanilla RNN both use a learning rate of $0.001$, and the LSTM utilizes 2 stacked layers. All networks use the Adam optimizer and their gradients clipped to 5. 

The dataset can be found in the following reference:
    Asseman, A., T. Kornuta, and A. Ozcan (2018) “Learning beyond simulated physics,” in Modeling and Decision-making in the Spatiotemporal Domain Workshop.
    URL https://openreview.net/forum?id=HylajWsRF7

How to execute code:

The DSRNN cell can be found in the file "c_k_rnn.py", which is used to design the DSRNN which is run and trained in the file "main_DSRNN.py". All supplemental functions used can be found in the file "sup_functions.py". 
