This repository contains implementations for the dynamically stabilized recurrent neural network (DSRNN) proposed in "A Dynamically Stabilized Recurrent Neural Network" in Tensorflow.

This novel recurrent neural network (RNN) architecture includes learnable skip-connections across a specified number of time-steps, which allows for a state-space representation of the network’s hidden-state trajectory, and introduces a regularization term in the loss function by utilizing the Lyapunov stability theory. The regularizer enables placement of eigenvalues of the (linearized) transfer function matrix to desired locations in the complex plane, thereby acting as an internal controller for the hidden-state trajectories.

# Experiment

The dataset used to validate the DSRNN is of a recorded double pendulum experiment, which is a chaotic system that exhibits behavior of long-term unpredictability. The results show that the DSRNN outperforms both the Long Short-Term Memory (LSTM) and vanilla recurrent neural networks, and the relative mean-squared error of the LSTM is reduced by up to ~99.64%. 

![visual_of_joints_trajectories](https://user-images.githubusercontent.com/44982976/122837944-050b1480-d2c3-11eb-8909-d89ff71a298a.png)
The figure above is a visualization of the double pendulum joints' trajectories at three random instances, each with a time-series length of 10 steps. Each joint is represented by a black circle, where the direction of travel is illustrated as travelling from the lightly shaded joints to the darker ones. The five black cross-markers, shown after each moving joint, represent the true positions of the joints, T, {1, 5, 10, 15, 20} time-steps ahead. The value of each T is marked, where the red numbers correspond to the second (middle) joint, and the green numbers correspond to the third (last) joint.

The networks were chosen after a basic hyperparameter search, where the batch size, learning rate, and number of LSTM blocks are varied. Batch sizes of {100, 250, 500, 1000}, learning rates of {0.1, 0.01, 0.001, 0.0001}, and LSTMs with 1 and 2 stacked layers are all considered. All networks contain 128 hidden units and the weight matrices of the DSRNN models are initialized with Glorot initializers. All the networks have a batch size of 250. The DSRNN models have a learning rate of 0.0001. The LSTM and vanilla RNN both use a learning rate of 0.001, and the LSTM utilizes 2 stacked layers. All networks use the Adam optimizer and their gradients clipped to 5. 

![Forecasting_errors](https://user-images.githubusercontent.com/44982976/122838307-cf1a6000-d2c3-11eb-920c-ac7612028268.png)
The figure above shows the average test errors over the five independent runs returned by each network model as the number of steps to forecast is increased, with the standard deviations across each five network models indicated by the error bars.

![visual_of_errors_per_joint](https://user-images.githubusercontent.com/44982976/122838319-d5a8d780-d2c3-11eb-8901-807a72de69f3.png)
The figure above is a visualization of the predicted locations of the dynamic joints of the double pendulum at three different instances by the DSRNN with k=3 and the LSTM. The highlighted regions corresponding to each recurrent network architecture are centered around the average prediction for each joint, corresponding to every {1, 5, 10, 15, 20} time-step prediction, T. The height and width of the circular regions are proportional in magnitude to 1 standard deviation of the x and y error predictions.

# Dataset

The dataset can be found in the following reference:
    Asseman, A., T. Kornuta, and A. Ozcan (2018) “Learning beyond simulated physics,” in Modeling and Decision-making in the Spatiotemporal Domain Workshop.
    URL https://openreview.net/forum?id=HylajWsRF7

# How to Execute Codes

The DSRNN cell can be found in the file "dsrnn.py", which is used to design the DSRNN in file "DSRNN_class.py", which is run and trained in the file "main_DSRNN.py". All supplemental functions used can be found in the file "sup_functions.py". 

To run this code, simply run the "main_DSRNN.py" file. Make sure datasets are saved in the directories of your choice, then update the directories of inside the script accordingly. Make sure to create files for the saved checkpoints.

You will need to specify the dataset being used; "DP" for double pendulum. The code right not can accomodate MNIST if dataset is set to "MNIST". Note: The LSTM results are not reported in the paper.
