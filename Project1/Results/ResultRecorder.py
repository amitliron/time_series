import csv
import numpy as np

file_name = ""

def create_headers(header, parameters):

    global file_name
    file_name= header + ".csv"
    with open(file_name, 'w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["", "", header])
        line = []
        line.append("test name")
        if parameters is not None:
            for param in parameters:
                line.append(param)
        line.append("score")
        writer.writerow(line)
    None

def record_results(test_name, params_dict, score_func, score_value, class_0, class_1):

    global file_name

    values_list = []
    values_list.append(test_name)
    values_list.append(params_dict)
    values_list.append(score_func)
    values_list.append(score_value)
    values_list = values_list + list(class_0)
    values_list.append("***")
    values_list = values_list + list(class_1)

    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(values_list)

    print(values_list)

def record_best_results(test_name, best_results):
    with open(file_name, 'a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([])
        writer.writerow([test_name, best_results])
