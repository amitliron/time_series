import skopt
# !pip install scikit-optimize if  necessary
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer
import multiprocessing

space = [
    Integer(low=500, high=5000, name='batch_size'),
    Integer(low=1, high=5, name='num_layers')
]
default_parameters = [60, 500, 1]

def worker(params, return_dict):
    print('timeSteps: ', params['timeSteps'])
    ret =  params['timeSteps']+3
    return_dict['ret'] = ret
    return

@use_named_args(space)
def objective(**params):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    p = multiprocessing.Process(target=worker, args=(params, return_dict))
    p.start()
    p.join()
    print(return_dict.values())
    return -return_dict.values()[0]


if __name__ == '__main__':
    gp_result = gp_minimize(func=objective,
                            dimensions=space,
                            n_calls=15,
                            noise=0.01,
                            n_jobs=-1,
                            kappa=4,
                            x0=default_parameters,
                            verbose=True)



