"""
    This script checks connectivity to the available GPU
"""

from keras import backend as K
K.tensorflow_backend._get_available_gpus()

import tensorflow as tf
config = tf.ConfigProto()  
config.gpu_options.allow_growth=True  
sess = tf.Session(config=config)  
