# import the necessary packages
import os
import argparse
import random
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import image, pyplot as plt
from random import sample,randint
matplotlib.use("Agg")

from preprocessor import preprocessor
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from sklearn.metrics import roc_curve, auc 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from unet import get_unet 

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.models import Model
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.python.keras.layers import Input, Activation, Reshape, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, BatchNormalization, GlobalAveragePooling2D
import wandb
from wandb.keras import WandbCallback

# initialize the number of epochs to train for, initial learning rate,
# batch size, and image dimensions

# set parameters
defaults=dict(
    learn_rate = 0.001,
    batch_size = 256,
    epochs = 100,
    )
wandb.init(project="master_thesis", config=defaults, name="unet_mask4_100samples")
config = wandb.config

#load data
f = open("/var/scratch/nsc400/hera_data/HERA_masks29-07-2020.pkl","rb")
dataset = pickle.load(f,encoding='latin1')
data = dataset[0]
labels = dataset[2]
mask1 = dataset[4]
mask2 = dataset[5]
mask3 = dataset[6]
mask4 = dataset[7]


d_height = data.shape[1]
d_width = data.shape[2]

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing

trainX, testX, trainY, testY = train_test_split(data,
 mask4, train_size=0.004, random_state=42)

# initialize the model using a sigmoid activation as the final layer

print("[INFO] compiling model...")
input_data = Input((d_height, d_width, 1), name='data')
model = get_unet(input_data, n_filters=16, dropout=0.05, batchnorm=True)

# initialize the optimizer
opt = Adam(lr=config.learn_rate,decay = config.learn_rate/config.epochs)
#decay = config.learn_rate/config.epochs
#opt = SGD(lr=config.learn_rate)
#opt = RMSprop(lr=config.learn_rate)

model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#print("[INFO] summary of model...")
#print(model.summary())

callbacks = [
    WandbCallback(),
    EarlyStopping(patience=50, verbose=1, monitor='val_loss'),
    ReduceLROnPlateau(factor=0.1, patience=30, verbose=1),
    ModelCheckpoint('model-unet-mask1-100.h5', verbose=1, save_best_only=True,save_weights_only=False)
]

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, batch_size=config.batch_size,
    validation_data=(testX, testY), epochs=config.epochs, verbose=1, callbacks=callbacks)

# log the number of total parameters
config.total_params = model.count_params()
print("Total params: ", config.total_params)

# save the model to disk
print("[INFO] serializing network...")
model.save("model_unet_mask4_100")

#save model
wandb.save('model_unet_rfi_impulse.h5')

# Predict on train, val and test
preds_train = model.predict(trainX, verbose=1)
preds_val = model.predict(testX, verbose=1)

preds_train_t = (preds_train > 0.5).astype(np.uint8)
preds_val_t = (preds_val > 0.5).astype(np.uint8)

#cf = ClassificationReport(ix)
#cf_mean = cf.generate(trainY, preds_train_t)
#print("Classification report mean : {}".format(cf_mean))
#classification report
#print(classification_report(testY, preds_val))

print('Classification report:\n', classification_report(testY.flatten(), preds_val_t.flatten()))

def plot_io(model,data,mask):

    mask = mask 
    output  = model.predict(data)
    binaryp = (output >0.04).astype(np.uint8)
    print(model.evaluate(data, mask, verbose=1))
    it = 1
    if isinstance(data,list):
        it = 2
        shape = output[0].shape[0]
    else:
        shape = output.shape[0]

    for i in range(it):
        fig,axs = plt.subplots(3,2,figsize=(10,10))

        if isinstance(data,list):
            inp = data[i]
            msk = mask[i]
            outp = output[i]
            bp = binaryp[i]
        else:
            inp = data
            msk = mask
            outp = output
            bp = binaryp

        for j in range(2):
            r = randint(0,shape-1)
            has_mask = msk[r,...,0].max() > 0

            axs[0,j].imshow(inp[r,...,0]);
            #if has_mask:
                #axs[0,j].contour(msk[r,...,0].squeeze(), levels=[0.1])
            axs[0,j].set_title(f' {labels[r]}',fontsize=10)

            axs[1,j].imshow(msk[r,...,0].squeeze(), vmin=0, vmax=1);
            axs[1,j].title.set_text('Mask {}'.format(r))

            #axs[2,j].imshow(outp[r,...,0]);
            #if has_mask:
                #axs[2,j].contour(msk[r,...,0].squeeze(),levels=[0.1])
            #axs[2,j].title.set_text('Mask Predicted{}'.format(r))

            axs[2,j].imshow(bp[r,...,0].squeeze(), vmin=0, vmax=1);
            if has_mask:
                axs[2,j].contour(msk[r,...,0].squeeze(),levels=[0.09])
            axs[2,j].title.set_text('Mask Binary Predicted{}'.format(r))


        return plt


wandb.log({'Analysis':plot_io(model,testX,testY)})


realm=testY.ravel()
predicted=preds_val.ravel()
fpr, tpr, _ = roc_curve(realm, predicted)
roc_auc = auc(fpr,tpr)

fig, ax = plt.subplots(1,1)
ax.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
ax.plot([0, 1], [0, 1], 'k--')
ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.legend(loc="lower right")
plt.grid()
plt.savefig('rocunet_mask4_100.png')
wandb.Image(plt)
wandb.log({"ROC": plt})
