# test multiple test cases in one running
# Add validation section in the training process, and test for the validation section

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
randomseed = 21
seed(randomseed)


import winsound
model_num = 5

def main():
    # load training data
    
    import scipy.io
    x = scipy.io.loadmat('../../DensityPrediction/DensityPredict-master/results-manually-saved/paper3_globalModel/Nor_GPinput_StormMonth_Without_HS_CA.mat')
    x = x['Nor_X'] # array

    y = scipy.io.loadmat('../../DensityPrediction/DensityPredict-master/results-manually-saved/paper3_globalModel/Nor_GPoutput_StormMonth_Without_HS_CA.mat')
    y = y['Nor_Y']

    n = len(x)
    # training_percentage = 0.7 # 1 as original data
    # training_number = round(n * training_percentage)
    # x = x[:training_number, :]
    # y = y[:training_number, :]
    validation_percentage = 0.2
    
    # training_number = round(n * validation_percentage)
    # train_x, validation_x = x[:training_number, :], x[training_number:, :]
    # train_y, validation_y = y[:training_number, :], y[training_number:, :]

    train_x, validation_x, train_y, validation_y = train_test_split(x, y, test_size=validation_percentage, random_state = randomseed, shuffle = True)

    input_dimension = train_x.shape[1]

    # load test case -1
    xt_1 = scipy.io.loadmat('../../DensityPrediction/DensityPredict-master/results-manually-saved/paper3_globalModel/Nor_GPinputC_03_1027_1103_CA.mat')
    xt_1 = xt_1['Nor_XT']

    yt_1 = scipy.io.loadmat('../../DensityPrediction/DensityPredict-master/results-manually-saved/paper3_globalModel/Nor_GPoutputC_03_1027_1103_CA.mat')
    yt_1 = yt_1['Nor_YT']

    # load test case - 2
    xt_2 = scipy.io.loadmat('../../DensityPrediction/DensityPredict-master/results-manually-saved/paper3_globalModel/Nor_GPinputA_03_1027_1103_CA.mat')
    xt_2 = xt_2['Nor_XT']

    yt_2 = scipy.io.loadmat('../../DensityPrediction/DensityPredict-master/results-manually-saved/paper3_globalModel/Nor_GPoutputA_03_1027_1103_CA.mat')
    yt_2 = yt_2['Nor_YT']
  
    # Define our model with an evidential output

    evidential_variable = 1e-2
    optimizer_variable = 2e-4
    dense_1_variable = 256

    model = tf.keras.models.Sequential()
    model.add(tf.keras.layers.Dense(dense_1_variable, activation='relu', input_dim=train_x.shape[1]))
    model.add(tf.keras.layers.Dense(64, activation='relu'))
    model.add(tf.keras.layers.Dense(512, activation='relu'))
    model.add(tf.keras.layers.Dense(256, activation='relu'))
    model.add(edl.layers.DenseNormalGamma(1))    # Output layer with single neuron for regression


    # Custom loss function to handle the custom regularizer coefficient
    def EvidentialRegressionLoss(true, pred):
        return edl.losses.EvidentialRegression(true, pred, coeff=evidential_variable)

    # Compile and fit the model!
    model.compile(
        # optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.1),
        optimizer=tf.keras.optimizers.Adam(optimizer_variable),
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
            model.add(tf.keras.layers.Dense(dense_1_variable, activation='relu', input_dim=train_x.shape[1]))
            model.add(tf.keras.layers.Dense(64, activation='relu'))
            model.add(tf.keras.layers.Dense(512, activation='relu'))
            model.add(tf.keras.layers.Dense(256, activation='relu'))
            model.add(edl.layers.DenseNormalGamma(1)) 
            model.compile(
            optimizer=tf.keras.optimizers.Adam(optimizer_variable),
            # optimizer = tf.keras.optimizers.experimental.SGD(learning_rate=0.1), 
            loss=EvidentialRegressionLoss)
            # model.fit(train_x, train_y, batch_size=128, epochs=250)
            
            # model.summary()

            EarlyStop = EarlyStopping(monitor='val_loss', patience=15, mode='min', restore_best_weights=True)
            model.fit(train_x, train_y, batch_size=128, epochs=300, validation_data = (validation_x,validation_y), callbacks=[EarlyStop], verbose = 1)
            
            model_weight_save_path = './EvidentialModel/TrainOnCA_New/Model_' + str(model_num) + '_ModelWeight_' + str(T)
            
            model.save_weights(model_weight_save_path)
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
    model.summary()
    mu_1, v_1, beta_1, var_1, sigma_1, mu_2, v_2, beta_2, var_2, sigma_2, mu_v, v_v, beta_v, var_v, sigma_v = predictor(model, xt_1, xt_2, validation_x, T = 3)
    

    import scipy.io
    mu_1 = np.array(mu_1)
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/mu_1.mat',
                {'mu_1':mu_1})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/v_1.mat',
                    {'v_1':v_1})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/beta_1.mat',
                    {'beta_1':beta_1})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/var_1.mat',
                    {'var_1':var_1})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/sigma_1.mat',
                    {'sigma_1':sigma_1})


    mu_v = np.array(mu_v)
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/mu_v.mat',
                {'mu_v':mu_v})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/v_v.mat',
                    {'v_v':v_v})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/beta_v.mat',
                    {'beta_v':beta_v})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/var_v.mat',
                    {'var_v':var_v})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/sigma_v.mat',
                    {'sigma_v':sigma_v})

    import scipy.io
    mu_2 = np.array(mu_2)
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/mu_2.mat',
                {'mu_2':mu_2})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/v_2.mat',
                    {'v_2':v_2})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/beta_2.mat',
                    {'beta_2':beta_2})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/var_2.mat',
                    {'var_2':var_2})
    scipy.io.savemat('../../DensityPrediction/DensityPredict-master/results-manually-saved/TrainOnCA_New/sigma_2.mat',
                    {'sigma_2':sigma_2})


    duration = 2000  # milliseconds
    freq = 440  # Hz
    winsound.Beep(freq, duration)

    



if __name__ == "__main__":
    main()