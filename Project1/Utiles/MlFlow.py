import time
import uuid
import os
import numpy as np
import pandas as pd
import shutil
import __main__
import inspect, os
import traceback
import logging


def ToCsv(_path):
    path = _path + 'MlFloew.txt'
    f_in = open(path, "r")
    lines = f_in.readlines()
    f_in.close()

    UniqueParameters = list()
    UniqueMetrics = list()

    IdDic = {}
    for line in lines:
        line_params = line.split(',')

        id = line_params[1].split(':')[1]
        date = line_params[2].split(':')[1]
        time = (line_params[3][7:]).strip()
        experiment = line_params[4].split(':')[1]
        source = line_params[5].split(':')[1]

        pram = ''
        metr = ''
        x = (line_params[6].split(':')[0]).strip()
        y = (line_params[6].split(':')[1]).strip()
        if x.startswith('pram_'):
            pram = x[5:]
        else:
            if x.startswith('metr_'):
                metr = x[5:]

        key = id + '_' + experiment
        if key not in IdDic:
            IdDic[key] = [[id, date, time, experiment, source], {}, {}]

        if pram != '':
            IdDic[key][1][pram] = y
            if pram not in UniqueParameters:
                UniqueParameters.append(pram)
        else:
            if metr != '':
                IdDic[key][2][metr] = y
                if metr not in UniqueMetrics:
                    UniqueMetrics.append(metr)

    UniqueParameters.sort()
    UniqueMetrics.sort()

    # --- create csv file -----
    allData = list()
    for key in IdDic:
        row = []
        row = np.append(row, IdDic[key][0])
        for pram in UniqueParameters:
            if pram in IdDic[key][1]:
                v = IdDic[key][1][pram]
                row = np.append(row, v)
            else:
                row = np.append(row, '---')

        for metr in UniqueMetrics:
            if metr in IdDic[key][2]:
                v = IdDic[key][2][metr]
                row = np.append(row, v)
            else:
                row = np.append(row, '---')

        allData.append(row)

    cols = []
    cols = np.append(cols, ['Id', 'Date', 'Time', 'Experiment', 'Source'])
    cols = np.append(cols, UniqueParameters)
    cols = np.append(cols, UniqueMetrics)

    path_csv = _path + 'MlFloew.csv'
    if os.path.isfile(path_csv):
        os.remove(path_csv)
    with open(path_csv, 'a') as file:
        ds = pd.DataFrame(allData, columns=cols)
        ds.to_csv(path_csv, index=False)

class MlFlow():
    def __init__(self, path='MlFlowData/xyz/', active=True):
        self.active = active
        self.date = time.strftime("%d/%m/%Y")
        self.time = time.strftime("%H:%M:%S")
        self.runId = uuid.uuid1()
        self.experiment = 0
        self.ver = 1.0

        # --- Get path --------
        self.set_Path(path)
        logging.basicConfig(filename= self.path + 'MlFlow.log', level=logging.DEBUG)
        # ----- open file for project -----
        self.file = open(self.path + 'MlFloew.txt', 'a')

    def __del__(self):
        self.file.close()

    # ------ geters ---------
    # def getPath(self, depth):
    #     sourcePath = (__main__.__file__)
    #     pathSplit = sourcePath.split('/')
    #     split_len = len(pathSplit)
    #     trimSize = 0
    #     for i in range(1,depth+2):
    #         trimSize += len(pathSplit[split_len - i])
    #     trimSize += depth
    #     path = sourcePath[:-trimSize]
    #     path += 'MlFlowData/'
    #     if not os.path.exists(path):
    #         os.makedirs(path)
    #     return path

    def getSource(self):
        stackLines = traceback.format_stack()
        s = len(stackLines)
        source = '----------'
        while s-1 >= 0:
            if not ('MlFlow.py' in stackLines[s-1]):
                path = ((stackLines[s - 1].split(',')[0]).strip())
                source = (path.split('\\')[-1])[:-1]
                break
            s -= 1

        return source

    def get_header(self):
        source = self.getSource()
        header = 'Var : ' + str(self.ver) + ' , Id : ' + str(
            self.runId) + ' , Date : ' + self.date + ' , Time : ' + self.time + ' , Experiment : ' + str(
            self.experiment) + ' , Source : ' + source + ' , '
        return header

    # ------ seters ---------
    def set_Path(self,path):
        self.path = path
        if not os.path.exists(path):
            os.makedirs(path)

    def set_runId(self,runId):
        self.runId = runId

    def set_experiment(self,experiment):
        self.experiment = experiment

    # ---------- Runable ---------
    def log_error(self,msg):
        print(msg)
        logging.error(time.strftime(msg + " %d/%m/%Y %H:%M:%S"))

    def NewExperiment(self):
        if self.active:
            self.experiment += 1

    def log_parameter(self, name, value):
        if self.active:
            try:
                logStr = self.get_header() + 'pram_' + name + ' : ' + str(value)
                self.file.write(logStr + '\n')
                self.file.flush()
            except Exception as e:
                self.log_error('Failed to log_parameter: ' + str(e))

    def log_metric(self, name, value):
        if self.active:
            try:
                logStr = self.get_header() + 'metr_' + name + ' : ' + str(value)
                self.file.write(logStr + '\n')
                self.file.flush()
            except Exception as e:
                self.log_error('Failed to log_metric: ' + str(e))



    def log_files(self,filesList):
        if self.active:
            try:
                path = self.path + self.projectName + '/' + str(self.runId) + '/'
                if not os.path.exists(path):
                    os.makedirs(path)

                for source in filesList:
                    shutil.copy(source, path)

            except Exception as e:
                self.log_error('Failed to log_files: ' + str(e))

    def log_pltFig(self,pltFig, pltFigName):
        if self.active:
            try:
                path = self.path + 'LogPltFig/' + self.projectName + '/' + str(self.runId) + '/'
                if not os.path.exists(path):
                    os.makedirs(path)

                plt.savefig(path + pltFigName + '.png')

            except Exception as e:
                self.log_error('Failed to log_files: ' + str(e))



