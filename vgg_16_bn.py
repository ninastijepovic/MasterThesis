import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Activation, Reshape, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, BatchNormalization, SpatialDropout2D, GlobalAveragePooling2D

def model_VGG16_BN(Base, img_ch, img_width, img_height, classes):
    model_VGG16_BN = Sequential()
    model_VGG16_BN.add(Conv2D(Base, input_shape=(img_height, img_width, img_ch), kernel_size=(3, 3), padding='same',
                     activation='relu'))
    model_VGG16_BN.add(Conv2D(Base, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(BatchNormalization(momentum = 0.01, epsilon = 0.9))
    model_VGG16_BN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model_VGG16_BN.add(Conv2D(Base * 2, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(Conv2D(Base * 2, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(BatchNormalization(momentum = 0.01, epsilon = 0.9))
    model_VGG16_BN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model_VGG16_BN.add(Conv2D(Base * 4, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(Conv2D(Base * 4, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(Conv2D(Base * 4, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(BatchNormalization(momentum = 0.01, epsilon = 0.9))
    model_VGG16_BN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model_VGG16_BN.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(BatchNormalization(momentum = 0.01, epsilon = 0.9))
    model_VGG16_BN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model_VGG16_BN.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(Conv2D(Base * 8, (3, 3), activation='relu', padding='same'))
    model_VGG16_BN.add(BatchNormalization(momentum = 0.01, epsilon = 0.9))
    model_VGG16_BN.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    model_VGG16_BN.add(Flatten())
    model_VGG16_BN.add(Dense(Base * 8, activation='relu'))
    model_VGG16_BN.add(BatchNormalization(momentum = 0.01, epsilon = 0.9))
    model_VGG16_BN.add(Dropout(0.7))
    model_VGG16_BN.add(Dense((classes), activation='sigmoid'))

    model_VGG16_BN.summary()
    return model_VGG16_BN
