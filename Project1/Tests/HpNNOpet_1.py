from keras.datasets import mnist
import keras
from keras.models import Sequential
from keras.layers import Dense
import tensorflow
from tensorflow.python.keras import backend as K
from keras.optimizers import Adam

import skopt
# !pip install scikit-optimize if  necessary
from skopt import gbrt_minimize, gp_minimize
from skopt.utils import use_named_args
from skopt.space import Real, Categorical, Integer





(X_train, y_train), (X_test, y_test) = mnist.load_data()

#Scale the data to between 0 and 1
X_train = X_train/ 255
X_test = X_test/ 255

#Flatten arrays from (28x28) to (784x1)
X_train = X_train.reshape(60000,784)
X_test = X_test.reshape(10000,784)

#Convert the y's to categorical to use with the softmax classifier
from keras.utils import np_utils
y_train = np_utils.to_categorical(y_train, 10)
y_test = np_utils.to_categorical(y_test, 10)

#Establish the input shape for our Networks.
input_shape= X_train[0].shape


print('###################################### Opet ######################################################')
space = [Real(low=1e-4, high=1e-2, prior='log-uniform',name='learning_rate'),
              Integer(low=1, high=5, name='num_dense_layers'),
              Integer(low=1, high=512, name='num_input_nodes'),
              Integer(low=1, high=28, name='num_dense_nodes'),
              Categorical(categories=['relu', 'sigmoid'],name='activation'),
              Integer(low=50, high=128, name='batch_size'),
              Real(low=1e-6, high=1e-2, name="adam_decay")
             ]
default_parameters = [1e-3, 1,512, 13, 'relu',64, 1e-3]


def create_model(**params):
    # start the model making process and create our first layer
    model = Sequential()
    model.add(Dense(params['num_input_nodes'], input_shape=input_shape, activation=params['activation']))
    # setup our optimizer and compile
    adam = Adam(lr=params['learning_rate'], decay=params['adam_decay'])
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

@use_named_args(space)
def objective(**params):
    model = create_model(**params)

    blackbox = model.fit(x=X_train,
                         y=y_train,
                         epochs=6,
                         batch_size=params['batch_size'],
                         validation_split=0.15,
                         verbose=1
                         )

    # return the validation accuracy for the last epoch.
    accuracy = blackbox.history['val_acc'][-1]

    # Print the classification accuracy.
    print()
    print("Accuracy: {0:.2%}".format(accuracy))
    print()

    # Delete the Keras model with these hyper-parameters from memory.
    del model

    # Clear the Keras session, otherwise it will keep adding new
    # models to the same TensorFlow graph each time we create
    # a model with a different set of hyper-parameters.
    K.clear_session()
    tensorflow.reset_default_graph()

    # the optimizer aims for the lowest score, so we return our negative accuracy
    return -accuracy


gp_result = gp_minimize(func=objective,
                        dimensions=space,
                        n_calls=15,
                        noise=0.01,
                        n_jobs=-1,
                        kappa=4,
                        x0=default_parameters,
                        verbose=True)

print('best result')
print(gp_result.x[0], gp_result.x[1], gp_result.x[2], gp_result.x[3], gp_result.x[4], gp_result.x[5])
model = create_model(gp_result.x[0], gp_result.x[1], gp_result.x[2], gp_result.x[3], gp_result.x[4], gp_result.x[5])
model.fit(X_train, y_train, epochs=3)
model.evaluate(X_test, y_test)

print('End opt')