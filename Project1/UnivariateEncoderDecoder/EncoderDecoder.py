from Utiles import SimulatedData as sd
from random import randint
from numpy import array
from numpy import argmax
from numpy import array_equal
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Dense, LSTM, Input
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error
import os
import multiprocessing
import time

import skopt
# !pip install scikit-optimize if  necessary
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

import Utiles.MlFlow as mlFlow
import uuid

from Features import HistFeatures as hs, SeriesFeatures as sf

varName = 'diff'
epochs = 2
verbose = 1
seed = 7
test_size = 0.33
bNormalize = False
n_features = 1
n_output = 2

space = [
    Categorical(categories=[30,60, 120, 200, 400, 600], name='timeSteps'),
    Integer(low=200, high=2000, name='batch_size'),
    Integer(low=50, high=1500, name='num_units'),
    Categorical(categories=['relu', 'elu', 'sigmoid', 'softmax', 'softplus', 'softsign', 'tanh', 'selu', 'exponential', 'linear'], name='activation'),
]
default_parameters = [60, 300, 100, 'relu']

def create_model(runParmsDic, **params):
    # define training encoder
    encoder_inputs = Input(shape=(None, params['timeSteps']))
    encoder = LSTM(params['num_units'], activation=params['activation'], return_state=True)
    encoder_outputs, state_h, state_c = encoder(encoder_inputs)
    encoder_states = [state_h, state_c]
    # define training decoder
    decoder_inputs = Input(shape=(None, n_output))
    decoder_lstm = LSTM(params['num_units'], activation=params['activation'], return_sequences=True, return_state=True)
    decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states)
    decoder_dense = Dense(n_output, activation='softmax')
    decoder_outputs = decoder_dense(decoder_outputs)
    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)
    # define inference encoder
    encoder_model = Model(encoder_inputs, encoder_states)
    # define inference decoder
    decoder_state_input_h = Input(shape=(params['num_units'],))
    decoder_state_input_c = Input(shape=(params['num_units'],))
    decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
    decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs)
    decoder_states = [state_h, state_c]
    decoder_outputs = decoder_dense(decoder_outputs)
    decoder_model = Model([decoder_inputs] + decoder_states_inputs, [decoder_outputs] + decoder_states)
    # return all models
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model, encoder_model, decoder_model

def Predict(prd_seq, infenc, infdec, n_features, n_timesteps_X, n_timesteps_y, runParmsDic):
    # --- predict_sequence ----
    prd_seq = prd_seq.reshape((1, n_features, n_timesteps_X))

    # encode
    state = infenc.predict(prd_seq)

    # Generate empty target sequence of length 1.
    target_seq = np.zeros((1, runParmsDic['n_output']))
    target_seq = target_seq.reshape((1, n_features, n_timesteps_y))

    # Populate the target sequence with end of encoding series pageviews
    target_seq[0, 0, :] = prd_seq[0, 0, -n_output :]

    output, h, c = infdec.predict([target_seq] + state)

    output_z = output.reshape((1, n_timesteps_y))
    return output_z

def worker(params,runParmsDic,return_dict):
    print('######################### ' + os.path.basename(__file__) + ' ############################')
    mlFlow1 = mlFlow.MlFlow(path='./../MlFlowData/' + runParmsDic['projectName'] + '/', active=runParmsDic['active'])
    mlFlow1.set_experiment(uuid.uuid1())
    for pram, pramVal in params.items():
        mlFlow1.log_parameter(pram, pramVal)
        print(pram, ': ', pramVal)

    for pram, pramVal in runParmsDic.items():
        mlFlow1.log_parameter(pram, pramVal)
        print(pram, ': ', pramVal)

    train, infenc, infdec = create_model(runParmsDic, **params)
    print(train.summary())

    # ------ Univariate series ------
    start_time = time.time()
    print('Start train EncoderDecoder LSTM model ....')
    simulatedData = sd.GetSimulatData(runParmsDic['dsName'])
    X, y = sf.BuildDataSetForTimeSeries_Univariate(ds=simulatedData, steps=params['timeSteps'], varName=varName, bNormalize=bNormalize)
    y = to_categorical(y)
    # split data into train and test sety
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # ---- reshape -----
    n_samples_train = X_train.shape[0]
    n_samples_test = X_test.shape[0]
    n_timesteps_X = X.shape[1]
    n_timesteps_y = y.shape[1]
    n_features = 1

    X_train = X_train.reshape((n_samples_train, n_features, n_timesteps_X))
    y_train = y_train.reshape((n_samples_train, n_features, n_timesteps_y))

    X_test = X_test.reshape((n_samples_test, n_features, n_timesteps_X))
    y_test = y_test.reshape((n_samples_test, n_features, n_timesteps_y))

    # fit model
    # --- Fit ----
    earlyStopping = EarlyStopping(monitor='val_mean_absolute_error', min_delta=0, patience=10, verbose=0, mode='auto',
                                  baseline=None, restore_best_weights=True)
    callbacks_list = [earlyStopping]

    blackbox = train.fit([X_train, y_train], y_train, epochs=epochs, batch_size=params['batch_size'],
                        validation_data=([X_test, y_test], y_test), callbacks=callbacks_list, verbose=verbose)
    # return the validation accuracy for the last epoch.
    accuracy = blackbox.history['val_acc'][-1]

    # Print the classification accuracy.
    print("Accuracy: {0:.2%}".format(accuracy))

    seconds = (time.time() - start_time)
    print('End train model: ' + 'seconds= ' + str("%.3f" % seconds))

    mlFlow1.log_metric('seconds', seconds)
    mlFlow1.log_metric('acc', accuracy)

    return_dict['ret'] = -accuracy
    return

@use_named_args(space)
def objective(**params):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(target=worker, args=(params,runParmsDic,return_dict))
    p.start()
    p.join()
    return return_dict['ret']

def Run(_runParmsDic):
    global runParmsDic
    runParmsDic = _runParmsDic

    gp_result = gp_minimize(func=objective,
                            dimensions=space,
                            n_calls=50,
                            noise=0.01,
                            n_jobs=-1,
                            kappa=4,
                            x0=default_parameters,
                            verbose=True)


