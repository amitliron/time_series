from Results import ResultRecorder as result_recorder
from Utiles import SimulatedData as sd
from Features import HistFeatures as hs, SeriesFeatures as sf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import Utiles.HPUtile as hpUtl
import os

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from keras.utils import to_categorical
from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, LSTM, Bidirectional, Input, Flatten, Activation, RepeatVector, Permute, Lambda, \
    BatchNormalization
from keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping, ReduceLROnPlateau

import multiprocessing
import skopt
# !pip install scikit-optimize if  necessary
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer

import Utiles.MlFlow as mlFlow
import uuid

epochs = 200
verbose = 1
seed = 7
test_size = 0.33
bNormalize = False
n_features = 3

space = [
    Categorical(categories=[60, 120, 200, 400], name='timeSteps'),
    Integer(low=500, high=5000, name='batch_size'),
    Integer(low=1, high=3, name='num_layers'),
    Integer(low=20, high=100, name='num_units')
]
default_parameters = [60, 500, 1, 20]

def create_model(**params):
    # -------- Bidirectional  LSTM ---------------------
    input = Input(shape=(params['timeSteps'], n_features))
    # --- First layer ---
    if params['num_layers'] == 1:
        h = Bidirectional(LSTM(units=params['num_units'], activation='relu'))(input)
    else:
        h = Bidirectional(LSTM(units=params['num_units'], activation='relu', return_sequences=True))(input)

    # --- Midels layers ---
    for i in range(1, params['num_layers'] - 1):
        h = Bidirectional(LSTM(units=params['num_units'], activation='relu', return_sequences=True))(h)

    # --- Last layer ---
    if params['num_layers'] > 1:
        h = Bidirectional(LSTM(units=params['num_units'], activation='relu'))(h)

    out = Dense(units=2, activation='softmax')(h)
    model = Model(inputs=input, outputs=out)

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['acc'])
    return model

def worker(params,runParmsDic,return_dict):
    print('######################### ' + os.path.basename(__file__) + ' ############################')
    mlFlow1 = mlFlow.MlFlow(path='./../MlFlowData/' + runParmsDic['projectName'] + '/', active=runParmsDic['active'])
    mlFlow1.set_experiment(uuid.uuid1())
    mlFlow1.log_parameter('dsName', runParmsDic['dsName'])
    for pram, pramVal in params.items():
        mlFlow1.log_parameter(pram, pramVal)
        print(pram, ': ', pramVal)

    model = create_model(**params)
    print(model.summary())

    # ------ Univariate series ------
    start_time = time.time()
    print('Start train Univariate LSTM model ....')
    # ds = sd.GetDemoData1()
    ds = sd.GetSimulatData(runParmsDic['dsName'])
    X, y = sf.BuildDataSetForTimeSeries_Multivariate(ds=ds, steps=params['timeSteps'], bNormalize=bNormalize)
    # sd.PrintDs(X)
    y = to_categorical(y)

    # split data into train and test sety
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

    # No need for reshape
    # n_samples_train = X_train.shape[0]
    # n_samples_test = X_test.shape[0]

    n_timesteps = X.shape[1]
    # X_train = X_train.reshape((n_samples_train, n_timesteps, n_features))
    # X_test = X_test.reshape((n_samples_test, n_timesteps, n_features))

    # fit model
    earlyStopping = EarlyStopping(monitor='val_acc', min_delta=0, patience=10, verbose=0, mode='auto',
                                  baseline=None, restore_best_weights=False)
    reduceLROnPlateau = ReduceLROnPlateau(monitor='val_acc', factor=0.5, patience=10, verbose=0, mode='auto',
                                          min_delta=0.0001, cooldown=0, min_lr=0)
    callbacks_list = [earlyStopping, reduceLROnPlateau]
    blackbox = model.fit(X_train, y_train, epochs=epochs, verbose=verbose, batch_size=params['batch_size'],
              validation_data=(X_test, y_test), callbacks=callbacks_list)

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

    p = multiprocessing.Process(target=worker, args=(params, runParmsDic, return_dict))
    p.start()
    p.join()
    return return_dict['ret']


def Run(_runParmsDic):
    global runParmsDic
    runParmsDic = _runParmsDic

    # ------- accuracy ------
    result_recorder.create_headers(header="BLSTM_UNIVAR-accuracy", parameters=None)
    runParmsDic['optimzation_func'] = 'acc'
    gp_result = gp_minimize(func=objective,
                            dimensions=space,
                            n_calls=15,
                            noise=0.01,
                            n_jobs=-1,
                            kappa=4,
                            x0=default_parameters,
                            verbose=True)
    result_recorder.record_best_results("BLSTM_UNIVAR", gp_result)
    print("Best Results: ", gp_result)

    # ------- ks ------
    result_recorder.create_headers(header="BLSTM_UNIVAR-ks", parameters=None)
    runParmsDic['optimzation_func'] = 'ks'
    gp_result = gp_minimize(func=objective,
                            dimensions=space,
                            n_calls=15,
                            noise=0.01,
                            n_jobs=-1,
                            kappa=4,
                            x0=default_parameters,
                            verbose=True)
    result_recorder.record_best_results("BLSTM_UNIVAR", gp_result)
    print("Best Results: ", gp_result)

    # ------- finished ------
    print("Finished")

