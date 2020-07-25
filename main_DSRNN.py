import tensorflow as tf
import numpy as np
from tqdm import tqdm
import time
import os
import matplotlib.pyplot as plt

os.chdir('/Codes')
from supp_functions import get_batches, load_data
from ckRNN_class import ckRNN

gpu_num = "1"
os.environ["CUDA_VISIBLE_DEVICES"]=gpu_num

#%% Script Options

save_records    = True

save_traj_plot  = True
save_error_hist = True
save_loss_plot  = True

only_dynamic_joints = True # If False then also the fixed joint is considered, otherwise only consider the two moving joints

num_trials = 5

# Dataset related params
dataset = 'DP' 
forecast_steps = 20 # Choose either [1, 5, 10, 15, 20] for DP
seq_len = 10

#%% Directories

dir_data = '/double-pendulum-chaotic/train_and_test_split/dpc_dataset_traintest_4_200_csv/RNN_traintest/seqlen_'
dir_results = '/Results/DP'


#%% Parameters

learning_rate = 1e-4
batch_size    = 250
epochs        = 500
n_hidden      = 128       # number of hidden units
n_layers      = 1         # number of hidden layers used
keep_prob     = 1.0       # Dropout keep probability
c_n           = 3         # c_k_rnn, k parameter

#%% Load Data

X_train, X_test_temp, y_train_temp, y_test_temp, n_input, n_steps, num_classes = load_data(dataset, forecast_steps, dir_data, seq_len)

md_index_test = len(y_test_temp)//2
X_val = X_test_temp[:md_index_test,:,:]
X_test = X_test_temp[md_index_test:,:,:]

if only_dynamic_joints == True:
    
    y_train = y_train_temp[:,2:6]
    y_val = y_test_temp[:md_index_test,2:6]
    y_test = y_test_temp[md_index_test:,2:6]
    
    num_classes = 4    
else:
    y_train = y_train_temp
    y_val = y_test_temp[:md_index_test,:]
    y_test = y_test_temp[md_index_test:,:]


#%% Create DCRNN Model

model = ckRNN(num_classes, batch_size=batch_size, num_steps=n_steps, num_inputs=n_input,
                rnn_size=n_hidden, num_layers=n_layers,
                learning_rate=learning_rate, dataset_name=dataset, c_k=c_n)
graph = tf.get_default_graph()

#%% Train DCRNN

error_test = np.zeros((batch_size*(np.shape(y_test)[0]//batch_size),np.shape(y_test)[1],2)) # The [2] because we save error (0) and true (1)

saver = tf.train.Saver(max_to_keep=1000)

if save_records == True:
    os.chdir(dir_results)

config = tf.ConfigProto()
config.gpu_options.allow_growth=True

#new_state_temp = np.zeros((epochs,batch_size, n_hidden))

for i in tqdm(range(1,num_trials+1), desc="\nTraining progress"):
    
    # Records variables
    records = np.zeros([epochs,6])                # records keeping, nn_loss, reg_loss, train_acc, val_acc, val_loss, time per epoch
    records_per_joint = np.zeros([batch_size,int(len(y_test)/batch_size),num_classes*2]) # records keeping error_1,...,6, corr(true,pred)_1,...,6
    
    with tf.Session(config=config) as sess:
        
        sess.run(tf.global_variables_initializer())
        counter = 0
        
        save_data_direc_best = dir_results+"/best_train_dsrnn_model_k{}_hl{}_bs{}_lr{}_sl_{}_trial{}.ckpt".format(c_n, n_hidden, batch_size,learning_rate,forecast_steps,i)
        save_data_direc_train = dir_results+"/train_dsrnn_model_k{}_hl{}_bs{}_lr{}_sl_{}_trial{}.ckpt".format(c_n, n_hidden, batch_size,learning_rate,forecast_steps,i)
    
        if not os.path.exists(save_data_direc_best):
            os.makedirs(save_data_direc_best)
        if not os.path.exists(save_data_direc_train):
            os.makedirs(save_data_direc_train)
        
        for e in range(epochs):
            # Train network
            new_state = sess.run(model.initial_state)
            loss = 0
            sess.run(tf.local_variables_initializer())
            
            
            start = time.time()
            train_acc = 0
            train_loss_nn = 0
            train_loss_reg = 0
            for x, y in get_batches(X_train, y_train, n_steps, batch_size):
                counter += 1
#                start = time.time()
                feed = {model.inputs: x,
                        model.targets: y,
                        model.keep_prob: keep_prob,
                        model.initial_state: new_state}
                batch_loss_nn, batch_loss_reg, train_acc_now, new_state, _ = sess.run([model.loss_nn, model.regularizer,
                                                                                   model.accuracy_op,
                                                                                   model.final_state, 
                                                                                   model.optimizer], feed_dict=feed)
                train_acc += train_acc_now
                train_loss_nn  += batch_loss_nn
                train_loss_reg += batch_loss_reg
            
            train_acc = train_acc/(len(X_train)/batch_size)
            
            records[e,0] = train_loss_nn
            records[e,1] = train_loss_reg
            records[e,2] = train_acc
                
            end = time.time()    
            
            time_per_epoch = end-start
                # control the print lines
            if (e+1) % 1 == 0:
                sess.run(tf.local_variables_initializer())
    #            stream_vars = [i for i in tf.local_variables()]
    #            print('[total, count]:',sess.run(stream_vars))  #accuracy metrics
#                print('\n',
#                      'Epoches: {}/{}... '.format(e+1, epochs),
#                      'Training Steps: {}... '.format(counter),
#                      'Training Loss: {:.4f}... '.format(batch_loss_nn),
#                      'Regularizer Loss: {:.4f}... '.format(batch_loss_reg),
#                      'Training Accuracy: {:.4f}... '.format(train_acc),
#                      '{:.4f} sec/batch'.format((end-start)))
#                print("\n beta_reg=",graph.get_tensor_by_name("beta_reg:0").eval())
                
                counter_for_val = 0
                
                val_acc = 0
                val_loss = 0
                for X_val_rnn, y_val_rnn in get_batches(X_val, y_val, 
                                                          n_steps, batch_size):
                    feed = {model.inputs: X_val_rnn,
                            model.targets: y_val_rnn,
                            model.keep_prob: 1.,
                            model.initial_state: new_state}
                    pred, val_loss_now, new_state_val, val_acc_now = sess.run([model.prediction,
                                                    model.loss,
                                                    model.final_state,
                                                    model.accuracy_op], 
                                                    feed_dict=feed)
                    val_loss += val_loss_now
                    val_acc  += val_acc_now
                    
                    error_val = y_val_rnn - pred
                    for ii in range(num_classes):
                        records_per_joint[:,counter_for_val,ii] = error_val[:,ii]
                        records_per_joint[:,counter_for_val,num_classes+ii] = np.correlate(y_val_rnn[:,ii],pred[:,ii])
                    
                    counter_for_val += 1
                val_acc = val_acc/(len(X_val)/batch_size)    

                
                if (e+1) % 10 == 0:
                    print('\n', 
                          'Validation loss: {:.4f}... '.format(val_loss),
#                          'Validation Accuracy: {:4f}... '.format(val_acc),
                          'Epochs: {}'.format(e+1))
                
                if val_acc > max(records[:,3]):
                    os.chdir(save_data_direc_best)
                    saver.save(sess, "best_train_dsrnn_model_k{}_hl{}_bs{}_lr{}_sl_{}_trial{}.ckpt".format(c_n, n_hidden, batch_size,learning_rate,forecast_steps,i))
                
                records[e,3] = val_acc
                records[e,4] = val_loss
                records[e,5] = time_per_epoch
                
                os.chdir(save_data_direc_train)
                saver.save(sess, "train_dsrnn_model_k{}_hl{}_bs{}_lr{}_sl_{}_trial{}.ckpt".format(c_n, n_hidden, batch_size,learning_rate,forecast_steps,i))
    
#        saver.save(sess, "dsrnn_model_k{}_hl{}_bs{}_lr{}_trial{}.ckpt".format(c_n, n_hidden, batch_size,learning_rate,i))                       
        sess.run(tf.local_variables_initializer())
        test_acc = 0
        test_loss = 0
        test_counter = 0
        for X_test_rnn, y_test_rnn in get_batches(X_test, y_test, 
                                                  n_steps, batch_size):
            feed = {model.inputs: X_test_rnn,
                    model.targets: y_test_rnn,
                    model.keep_prob: 1.,
                    model.initial_state: new_state}
            pred, test_loss_now, new_state_test, test_acc_now = sess.run([model.prediction,
                                            model.loss,
                                            model.final_state,
                                            model.accuracy_op], 
                                            feed_dict=feed)
            test_loss += test_loss_now
            test_acc  += test_acc_now
            
            error_range = range(test_counter*batch_size,(test_counter+1)*batch_size)
            error_test[error_range,:,0] = y_test_rnn - pred
            error_test[error_range,:,1] = y_test_rnn
            
            test_counter += 1
            
#                new_state_temp[e,:,:] = new_state_test
        
        test_acc = test_acc/(len(X_test)/batch_size)
        print('Final test loss is: '+str(test_loss))
      
    if save_records == True:
        os.chdir(dir_results)
        np.save('records_predsteps_{}_k_{}_batch_size_{}_trial_{}'.format(forecast_steps,c_n,batch_size,i),records)
        np.save('records_joints_predsteps_{}_k_{}_batch_size_{}_trial_{}'.format(forecast_steps,c_n,batch_size,i),records_per_joint)
        np.save('test_records_predsteps_{}_k_{}_batch_size_{}_trial_{}'.format(forecast_steps,c_n,batch_size,i),error_test)
        np.save('test_loss_predsteps_{}_k_{}_batch_size_{}_trial_{}'.format(forecast_steps,c_n,batch_size,i),test_loss)
