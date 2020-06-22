from Utiles import SimulatedData as sd
from Features import HistFeatures as hs
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import uuid
from Features import SeriesFeatures as sf
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from xgboost import plot_importance
# import confusion_matrix_pretty_print as cmpp
import os
import Utiles.MlFlow as mlFlow
from Results import ResultRecorder as result_recorder

def Run(runParmsDic):

    result_recorder.create_headers(header="Histograms_XGBoost", parameters=None)

    # ------ parameter --------
    # bins = [10,20,30,40, 50,60,70, 100, 200]
    bins = [10, 20, 30,50,100]
    # ------------- Hist -------------
    for bin in bins:
        mlFlow1 = mlFlow.MlFlow(path='./../MlFlowData/' + runParmsDic['projectName'] + '/', active=runParmsDic['active'])
        mlFlow1.set_experiment(uuid.uuid1())
        mlFlow1.log_parameter('dsName', runParmsDic['dsName'])
        print('######################### ' + os.path.basename(__file__) + ' ############################')
        print('Bins: ', bin)
        mlFlow1.log_parameter('dsName', runParmsDic['dsName'])

        start_time = time.time()
        print('Start train model ....')

        mlFlow1.log_parameter('bins', bin)
        ds = sd.GetSimulatData(runParmsDic['dsName'])
        BinsFeatures_ds = hs.BuildDataSet(ds, bin)
        # ---- explore ----
        # sd.GetSatistic(ds)
        # hs.PlotFeaturesHists(BinsFeatures_ds)
        # sd.PlotDataSet(ds)
        # sd.PlotDataSetRec(ds)

        dataset = BinsFeatures_ds.values


        # split data into X and y
        X = dataset[:, 0:bin * 3]
        Y = dataset[:, bin * 3]

        # split data into train and test sets
        seed = 7
        test_size = 0.33
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size, random_state=seed)
        print('Train on: ', len(X_train),'Test on: ', len(X_test))

        # fit model no training data
        model = XGBClassifier()
        model.fit(X_train, y_train)


        # print(model)

        # make predictions for test data
        y_pred = model.predict(X_test)


        # get prediction
        predictions = [round(value) for value in y_pred]

        # evaluate predictions
        accuracy = accuracy_score(y_test, predictions)
        print("Accuracy: %.2f%%" % (accuracy * 100.0))

        seconds = (time.time() - start_time)
        print('End train model: ' + 'seconds= ' + str("%.3f" % seconds))

        mlFlow1.log_metric('seconds', seconds)
        mlFlow1.log_metric('acc', accuracy)

        result_recorder.record_results(test_name="Hist_XGBoost", params_dict={"Bins:", bin}, score_func="Accuracy", score_value=accuracy, class_0=[], class_1=[])



# -------------------------------------------------
'''

cols = list(BinsFeatures_ds.columns)[0:-1]
model.get_booster().feature_names = cols
xgb_fea_imp=pd.DataFrame(list(model.get_booster().get_fscore().items()),
columns=['feature','importance']).sort_values('importance', ascending=False)
print('',xgb_fea_imp)
# xgb_fea_imp.to_csv('xgb_fea_imp.csv')
plot_importance(model, )


cmpp.plot_confusion_matrix_from_data(y_test,y_pred, columns=[0,1])

'''
