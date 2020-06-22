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


print('###################################### Base ######################################################')
'''
model = Sequential()
model.add(Dense(16, input_shape=input_shape, activation='relu',name = 'input_layer'))
model.add(Dense(16, activation='relu', name="hidden_layer"))

model.add(Dense(10,activation='softmax',name="output_layer"))
model.compile(optimizer = 'adam', loss='categorical_crossentropy', metrics=["accuracy"])

#model.summary prints a description of the model
#model.summary()

#model history is stored as "blackbox".
blackbox = model.fit(X_train, y_train, batch_size=128, epochs =3, validation_split=.15)

accuracy = model.evaluate(X_test,y_test)[1]
print(accuracy)
'''
print('###################################### Opet ######################################################')

dim_learning_rate = Real(low=1e-4, high=1e-2, prior='log-uniform',
                         name='learning_rate')
dim_num_dense_layers = Integer(low=1, high=5, name='num_dense_layers')
dim_num_input_nodes = Integer(low=1, high=512, name='num_input_nodes')
dim_num_dense_nodes = Integer(low=1, high=28, name='num_dense_nodes')
dim_activation = Categorical(categories=['relu', 'sigmoid'],
                             name='activation')
dim_batch_size = Integer(low=50, high=128, name='batch_size')
dim_adam_decay = Real(low=1e-6,high=1e-2,name="adam_decay")

dimensions = [dim_learning_rate,
              dim_num_dense_layers,
              dim_num_input_nodes,
              dim_num_dense_nodes,
              dim_activation,
              dim_batch_size,
              dim_adam_decay
             ]
default_parameters = [1e-3, 1,512, 13, 'relu',64, 1e-3]


def create_model(learning_rate, num_dense_layers, num_input_nodes,
                 num_dense_nodes, activation, adam_decay):
    # start the model making process and create our first layer
    model = Sequential()
    model.add(Dense(num_input_nodes, input_shape=input_shape, activation=activation
                    ))
    # create a loop making a new dense layer for the amount passed to this model.
    # naming the layers helps avoid tensorflow error deep in the stack trace.
    for i in range(num_dense_layers):
        name = 'layer_dense_{0}'.format(i + 1)
        model.add(Dense(num_dense_nodes,
                        activation=activation,
                        name=name
                        ))
    # add our classification layer.
    model.add(Dense(10, activation='softmax'))

    # setup our optimizer and compile
    adam = Adam(lr=learning_rate, decay=adam_decay)
    model.compile(optimizer=adam, loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model


@use_named_args(dimensions=dimensions)
def fitness(learning_rate, num_dense_layers, num_input_nodes,
            num_dense_nodes, activation, batch_size, adam_decay):

    print('------------',learning_rate, num_dense_layers, num_input_nodes,
            num_dense_nodes, activation, batch_size, adam_decay)

    model = create_model(learning_rate=learning_rate,
                         num_dense_layers=num_dense_layers,
                         num_input_nodes=num_input_nodes,
                         num_dense_nodes=num_dense_nodes,
                         activation=activation,
                         adam_decay=adam_decay
                         )

    # named blackbox becuase it represents the structure
    blackbox = model.fit(x=X_train,
                         y=y_train,
                         epochs=6,
                         batch_size=batch_size,
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


gp_result = gp_minimize(func=fitness,
                        dimensions=dimensions,
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