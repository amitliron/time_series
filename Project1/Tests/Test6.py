import skopt
# !pip install scikit-optimize if  necessary
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
import multiprocessing

from keras.models import Sequential
from keras.layers import Dense
import numpy as np

space = [
    Integer(low=5, high=50, name='batch_size'),
    Integer(low=1, high=5, name='num_layers')
]
default_parameters = [5, 1]

def create_model(num_layers):
    model = Sequential()
    model.add(Dense(8, input_dim=2, activation='relu'))
    for i in range(0, num_layers):
        model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

def worker(params, return_dict):
    print('num_layers: ', params['num_layers'])

    X = [[1, 2], [2, 4], [4, 5], [3, 2], [3, 1]]
    y = [0, 0, 1, 1, 2]

    X = np.array(X)
    y = np.array(y)

    model = create_model(params['num_layers'])
    # fit the keras model on the dataset
    model.fit(X, y, epochs=5, batch_size=params['batch_size'])
    # evaluate the keras model
    _, accuracy = model.evaluate(X, y)
    print('Accuracy: %.2f' % (accuracy * 100))

    return_dict['ret'] = -accuracy
    return

@use_named_args(space)
def objective(**params):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(target=worker, args=(params, return_dict))
    p.start()
    p.join()
    print(return_dict.values())
    return return_dict['ret']


if __name__ == '__main__':
    gp_result = gp_minimize(func=objective,
                            dimensions=space,
                            n_calls=15,
                            noise=0.01,
                            n_jobs=-1,
                            kappa=4,
                            x0=default_parameters,
                            verbose=True)



