# test multiple test cases in one running
# Based on Deep_evidential_learning-DensityPrediction_Standardlization_3
# Add validation section in the training process, and test for the validation section
# Add new data after training a model, and compare whether the data is different

## Mainly test different network structure with more data points
from cProfile import label
import functools
import numpy as np
import matplotlib.pyplot as plt

import evidential_deep_learning as edl
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping
# import models
# import trainers

import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"


import scipy.io
from sklearn.model_selection import train_test_split
from numpy.random import seed
randomseed = 9
seed(randomseed)


import winsound


def main():
    # load training data
    
    import scipy.io
    x = scipy.io.loadmat('GPinputC.mat')
    x = x['Nor_X'] # array

    y = scipy.io.loadmat('GPoutputC.mat')
    y = y['Nor_Y']

    n = len(x)

    validation_percentage = 0.3
    train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=validation_percentage, random_state = randomseed, shuffle = True)

    input_dimension = train_x.shape[1]

    # load test case -1
    xt_1 = scipy.io.loadmat('GPinputC_05_0801_0802.mat')
    xt_1 = xt_1['Nor_XT']

    yt_1 = scipy.io.loadmat('GPoutputC_05_0801_0802.mat')
    yt_1 = yt_1['Nor_YT']

    # load test case - 2
    xt_2 = scipy.io.loadmat('GPinputC_Nor_03_1027_1103.mat')
    xt_2 = xt_2['Nor_XT']

    yt_2 = scipy.io.loadmat('GPoutputC_Nor_03_1027_1103.mat')
    yt_2 = yt_2['Nor_YT']
  
    # Define our model with an evidential output

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(256, activation='relu', input_dim=train_x.shape[1]))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    # model.add(tf.keras.layers.Dense(1024, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(tf.keras.layers.Dense(128, activation='relu'))
    model.add(edl.layers.DenseNormalGamma(1))  # Output layer with single neuron for regression


    # Custom loss function to handle the custom regularizer coefficient
    def EvidentialRegressionLoss(true, pred):
        return edl.losses.EvidentialRegression(true, pred, coeff=1e-2)

    # Compile and fit the model
    model.compile(
        # optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.1),
        optimizer=tf.keras.optimizers.Adam(5e-4),
        # optimizer = 'adam',
        loss=EvidentialRegressionLoss)
        
    def predictor(model, 
            X_test_1, X_test_2, X_validation, T):

        probs_1 = []
        probs_2 = []
        probs_v = []
        # cur_mu=[]
        # cur_v=[]
        # cur_alpha=[]
        # cur_beta=[]
        mu_1 = []
        mu_2 = []
        mu_v = []
        v_1 = []
        v_2 = []
        v_v = []
        beta_1 = []
        beta_2 = []
        beta_v = []
        var_1 = []
        var_2 = []
        var_v = []
        sigma_1 = []
        sigma_2 = []
        sigma_v = []
        for _ in range(T):
            import random
            seedvalue = T+3
            random.seed(seedvalue)
            ## Check the model structure above!!!
            tf.random.set_seed(seedvalue)
            tf.keras.backend.clear_session()
           
            model = tf.keras.models.Sequential()
            model.add(tf.keras.layers.Dense(256, activation='relu', input_dim=train_x.shape[1]))
            model.add(tf.keras.layers.Dense(512, activation='relu'))
            # model.add(tf.keras.layers.Dense(1024, activation='relu'))
            model.add(tf.keras.layers.Dense(256, activation='relu'))
            # model.add(tf.keras.layers.Dense(128, activation='relu'))
            model.add(tf.keras.layers.Dense(128, activation='relu'))
            model.add(edl.layers.DenseNormalGamma(1)) 
            model.compile(
            optimizer=tf.keras.optimizers.Adam(5e-4),
            loss=EvidentialRegressionLoss)
           

            EarlyStop = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
            model.fit(train_x, train_y, batch_size=128, epochs=100, validation_data = (validation_x,validation_y), callbacks=[EarlyStop], verbose = 0)
            
            cur_pred_1 = model.predict(X_test_1, verbose=0)
            cur_mu_1, cur_v_1, cur_alpha_1, cur_beta_1 = tf.split(cur_pred_1, 4, axis=-1)
            cur_var_1 = np.sqrt(cur_beta_1 / (cur_v_1 * (cur_alpha_1 - 1)))
            cur_var_1 = np.minimum(cur_var_1, 1e3)[:, 0]  # for visualization
            cur_sigma_1 = np.sqrt(cur_beta_1 / (cur_alpha_1 - 1))
            cur_sigma_1 = np.minimum(cur_sigma_1, 1e3)[:, 0]
            probs_1 += [cur_pred_1]
            cur_mu_1 = np.array(cur_mu_1)
            mu_1 += [cur_mu_1]
            v_1 += [cur_v_1]
            beta_1 += [cur_beta_1]
            var_1 += [cur_var_1]
            sigma_1 += [cur_sigma_1]


            # test 2 case
            cur_pred_2 = model.predict(X_test_2, verbose=0)
            cur_mu_2, cur_v_2, cur_alpha_2, cur_beta_2 = tf.split(cur_pred_2, 4, axis=-1)
            cur_var_2 = np.sqrt(cur_beta_2 / (cur_v_2 * (cur_alpha_2 - 1)))
            cur_var_2 = np.minimum(cur_var_2, 1e3)[:, 0]  # for visualization
            cur_sigma_2 = np.sqrt(cur_beta_2 / (cur_alpha_2 - 1))
            cur_sigma_2 = np.minimum(cur_sigma_2, 1e3)[:, 0]
            probs_2 += [cur_pred_2]
            cur_mu_2 = np.array(cur_mu_2)
            mu_2 += [cur_mu_2]
            v_2 += [cur_v_2]
            beta_2 += [cur_beta_2]
            var_2 += [cur_var_2]
            sigma_2 += [cur_sigma_2]


            T -= 1
            print(T)
            
        return mu_1, v_1, beta_1, var_1, sigma_1, mu_2, v_2, beta_2, var_2, sigma_2, mu_v, v_v, beta_v, var_v, sigma_v
        # return mu, v, beta, var, sigma
  
    mu_1, v_1, beta_1, var_1, sigma_1, mu_2, v_2, beta_2, var_2, sigma_2, mu_v, v_v, beta_v, var_v, sigma_v = predictor(model, xt_1, xt_2, validation_x, T = 2)
    

    import scipy.io
    mu_1 = np.array(mu_1)
    scipy.io.savemat(' mu_1.mat',
                {'mu_1':mu_1})
    scipy.io.savemat(' v_1.mat',
                    {'v_1':v_1})
    scipy.io.savemat(' beta_1.mat',
                    {'beta_1':beta_1})
    scipy.io.savemat(' var_1.mat',
                    {'var_1':var_1})
    scipy.io.savemat(' sigma_1.mat',
                    {'sigma_1':sigma_1})


    import scipy.io
    mu_2 = np.array(mu_2)
    scipy.io.savemat(' mu_2.mat',
                {'mu_2':mu_2})
    scipy.io.savemat(' v_2.mat',
                    {'v_2':v_2})
    scipy.io.savemat(' beta_2.mat',
                    {'beta_2':beta_2})
    scipy.io.savemat(' var_2.mat',
                    {'var_2':var_2})
    scipy.io.savemat(' sigma_2.mat',
                    {'sigma_2':sigma_2})

    
    # Retrain with new data
    model.compile(
            optimizer=tf.keras.optimizers.Adam(5e-4),
            loss=EvidentialRegressionLoss)
    
    # load new and old data
    x = scipy.io.loadmat(' GPinputC_Nor_more_0401_05.mat')
    x = x['Nor_X'] # array

    y = scipy.io.loadmat(' GPoutputC_Nor_more_0401_05.mat')
    y = y['Nor_Y']

    # model.fit(x,y, epochs = 100, batch_size = 128, verbose = 0)
    EarlyStop = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
    model.fit(x,y, batch_size=128, epochs=100, validation_data = (validation_x,validation_y), callbacks=[EarlyStop], verbose = 0)
            

    new_mu=[]
    new_v=[]
    new_alpha=[]
    new_beta=[]
    new_pred_1=[]

    new_pred_1 = model.predict(xt_1, verbose=0)
    new_mu_1, new_v_1, new_alpha_1, new_beta_1 = tf.split(new_pred_1, 4, axis=-1)
    new_var_1 = np.sqrt(new_beta_1 / (new_v_1 * (new_alpha_1 - 1)))
    new_var_1 = np.minimum(new_var_1, 1e3)[:, 0]  # for visualization
    new_sigma_1 = np.sqrt(new_beta_1 / (new_alpha_1 - 1))
    new_sigma_1 = np.minimum(new_sigma_1, 1e3)[:, 0]

    import scipy.io
    new_mu_1 = np.array(new_mu_1)
    scipy.io.savemat(' new_mu_1.mat',
                {'new_mu_1':new_mu_1})
    scipy.io.savemat(' new_v_1.mat',
                    {'new_v_1':new_v_1})
    scipy.io.savemat(' new_beta_1.mat',
                    {'new_beta_1':new_beta_1})
    scipy.io.savemat(' new_var_1.mat',
                    {'new_var_1':new_var_1})
    scipy.io.savemat(' new_sigma_1.mat',
                    {'new_sigma_1':new_sigma_1})


    duration = 2000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)

    



if __name__ == "__main__":
    main()