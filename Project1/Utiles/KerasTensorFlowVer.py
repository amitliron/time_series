import tensorflow
import keras
from tensorflow.python.client import device_lib
from keras import backend as K



print('keras', keras.__version__)
print('tensorflow', tensorflow.__version__)

print('tensorflow', device_lib.list_local_devices())
print('keras', K.tensorflow_backend._get_available_gpus())

