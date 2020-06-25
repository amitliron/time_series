import UnivariateSeriesModels.CnnLstm_univar as uniVarCnnLstm
import UnivariateSeriesModels.ConvLstm_univar as uniVarConvLstm
import UnivariateSeriesModels.Lstm_univar as uniVarLstm
import UnivariateSeriesModels.BLstm_univar as uniVarBLstm

import MultivariateSeriesModels.Lstm_multivar as multiVarLstm
import MultivariateSeriesModels.BLstm_multivar as multiVarBLstm
import MultivariateSeriesModels.CnnLstm_multivar as multiVarCnnLstm
import MultivariateSeriesModels.ConvLstm_multivar as multiVarConvLstm

import UnivariateEncoderDecoder.EncoderDecoder as uniVarEncoderDecoder

import Utiles.MlFlow as mlFlow
import HistModels.HistXGboost as histXgb
import logging
import time

logging.basicConfig(filename= './RunLog.log', level=logging.DEBUG)

def run_function(func, paramsDic):
    try:
        func(paramsDic)
    except Exception as e:
        msg = 'Failed to run_function: ' + str(e)
        logging.error(time.strftime(msg + " %d/%m/%Y %H:%M:%S"))

if __name__ == '__main__':
    runParamsDic = { 'projectName': 'EventClass_1',
                     'dsName': 'Row99_f=40',
                     'active': True,
                     'optimzation_func': 'acc'
    }
    # ----------- Hist  multi var------
    #run_function(histXgb.Run,runParamsDic)
    # ----------- Serias  uni var------
    #run_function(uniVarCnnLstm.Run, runParamsDic)   # amitli changes
    #run_function(uniVarConvLstm.Run, runParamsDic)
    #run_function(uniVarLstm.Run, runParamsDic)
    #run_function(uniVarBLstm.Run, runParamsDic)

    # ----------- Serias  multi var------
    run_function(multiVarLstm.Run,runParamsDic)
    # run_function(multiVarBLstm.Run, runParamsDic)
    # run_function(multiVarCnnLstm.Run,runParamsDic)
    #run_function(multiVarConvLstm.Run,runParamsDic)

    # ----------- EncoderDecoder  uni var------
    # run_function(uniVarEncoderDecoder.Run, runParamsDic)

    mlFlow.ToCsv('./../MlFlowData/' + runParamsDic['projectName'] + '/')
