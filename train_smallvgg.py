# import the necessary packages
import os
import itertools
import sys
import argparse
import random
import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib
from scipy import interp
from itertools import cycle
from scipy import interp
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
from matplotlib import image, pyplot as plt
from random import sample, randint

from preprocessor import preprocessor
from tqdm import tqdm_notebook, tnrange
from itertools import chain
from sklearn.model_selection import train_test_split
#from tensorflow.keras.utils import to_categorical
from smallvgg import SmallVGGNet
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix

import tensorflow as tf
from tensorflow import keras
from tensorflow.python.keras import layers
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.python.keras.optimizers import Adam
from tensorflow.python.keras.layers import Activation, Reshape, Dropout, Flatten, Conv2D, MaxPooling2D, Dense, BatchNormalization
import wandb
from wandb.keras import WandbCallback

# initialize the number of epochs to train for, initial learning rate(default of Adam optimizer),
# batch size, and image dimensions

#EPOCHS = 30
#INIT_LR = 1e-3
#BS = 32
IMAGE_DIMS = (32, 128, 2)

# set parameters
defaults=dict(
    learn_rate = 0.0001,
    batch_size = 256,
    epochs = 200,
    )
wandb.init(project="master_thesis", config=defaults, name="small_vgg_network_2k")
config = wandb.config

#load data
f = open("/var/scratch/nsc400/hera_data/HERA_02-05-2020.pkl","rb")
dataset = pickle.load(f,encoding='latin1')
data = dataset[0]
labels = dataset[2]

#prepocess data 
#resize
p = preprocessor(data)
p.interp(32,128)
p.get_magnitude_and_phase()
p.median_threshold()
p.minmax(per_baseline=True,feature_range=(0,1))
processed_data = p.get_processed_cube()


#split labels
label = [l.split('-') for l in labels]
labels = np.array(label,dtype=object)
as_list = [list(i) for i in labels]

# binarize the labels using scikit-learn multi-label
# binarizer implementation
print("[INFO] class labels:")
mlb = MultiLabelBinarizer()
ohe = mlb.fit_transform(as_list) # might need to add .astype(float)

# loop over each of the possible class labels and show them
for (i, label) in enumerate(mlb.classes_):
    print("{}. {}".format(i + 1, label))

label_to_class = len(mlb.classes_)

# partition the data into training and testing splits using 80% of
# the data for training and the remaining 20% for testing

trainX, testX, trainY, testY = train_test_split(processed_data,
 ohe, test_size=0.92, random_state=42)

# initialize the model using a sigmoid activation as the final layer
# in the network so we can perform multi-label classification
print("[INFO] compiling model...")
model = SmallVGGNet.build(
        width=IMAGE_DIMS[1], height=IMAGE_DIMS[0],
        depth=IMAGE_DIMS[2], classes=len(mlb.classes_),
        finalAct="sigmoid")

# initialize the optimizer
opt = Adam(lr=config.learn_rate)

# compile the model using binary cross-entropy rather than
# categorical cross-entropy -- this may seem counterintuitive for
# multi-label classification, but keep in mind that the goal here
# is to treat each output label as an independent Bernoulli
# distribution
model.compile(loss="binary_crossentropy", optimizer=opt,metrics=["accuracy"])

print("[INFO] summary of model...")
print(model.summary())

#callbacks
callbacks = [
    WandbCallback(),
    EarlyStopping(patience=100, monitor='val_loss', verbose=1),
    #ReduceLROnPlateau(factor=0.1, patience=0, min_lr=0.00001,  verbose=1),
    ModelCheckpoint('model_smallvgg_2k_final.h5', verbose=1, save_best_only=True,save_weights_only=False)
]

# train the network
print("[INFO] training network...")
H = model.fit(trainX, trainY, batch_size=config.batch_size,
    validation_data=(testX, testY), epochs=config.epochs, verbose=1,callbacks=callbacks)

# log the number of total parameters
config.total_params = model.count_params()
print("Total params: ", config.total_params)

# save the model to disk
print("[INFO] serializing network...")
model.save("model_smallvgg_final_2k")
# save the multi-label binarizer to disk
print("[INFO] serializing label binarizer...")
f = open("labelbinsmallvgg_final_2k", "wb")
f.write(pickle.dumps(mlb))
f.close()

#save model
wandb.save('model_smallvgg_final.h5')

#predict
thresh = 0.5
y_prob = model.predict(testX)
y_pred = np.array([[1 if i > thresh else 0 for i in j] for j in y_prob])
print(multilabel_confusion_matrix(testY, y_pred))

#classification report 
print(classification_report(testY, y_pred))
report = classification_report(testY, y_pred, output_dict=True)
df = pd.DataFrame(report).transpose()
df.to_csv('Classification_Report_smallvgg_final.csv', index= True)

#probabilities for each label
my_list = []
for i in y_prob:
    my_list.append("-")
    for (label, p) in zip(mlb.classes_, i):
        my_list.append("{}: {:.2f}%".format(label, p * 100))


#save txt file
with open("my_list_resize_final.txt", "w") as output:
    output.write(str(my_list))

#Model Accuracy
scores = model.evaluate(testX, testY, verbose=0)
print("Accuracy: %.2f%%" % (scores[1]*100))

#plot analysis
def plot_analysis(model,data,labels):

    output  = model.predict(data)
    titles= mlb.inverse_transform(labels)
    it = 1
    if isinstance(data,list):
        it = 2
        shape = output[0].shape[0]
    else:
        shape = output.shape[0]

    for i in range(it):
        fig,axs = plt.subplots(3,2,figsize=(20,10))
        fig.subplots_adjust(wspace=0.5, hspace=0.5)

        if isinstance(data,list):
            inp = data[i]
            outp = output[i]
        else:
            inp = data
            outp = output

        for i in range(3):
            r = randint(0,shape-1)
            #o = output[sample]

            axs[i,0].set_title(f' {titles[r]}',fontsize=10 )
            axs[i,0].imshow(inp[r,...,0])

            axs[i,1].bar(range(len(mlb.classes_)), outp[r])
            axs[i,1].set_xticks(range(len(mlb.classes_)))
            axs[i,1].set_xticklabels(["{}. {}".format(i+1, label) for (i, label) in enumerate(mlb.classes_)],rotation=45, fontsize=10)

        return plt

wandb.log({'Analysis':plot_analysis(model,testX,testY)})  


def plot_confusion_matrix(cm, classes, title, ax):

    ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > cm.max() / 2. else "black")

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks), ax.xaxis.set_ticklabels(classes)
    ax.set_yticks(tick_marks), ax.yaxis.set_ticklabels(classes)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Truth')
    ax.set_title(title)
    ax.grid(False)

def plot_ml_confusion_matrix(testY, y_pred, label_to_class, save_plot=False):
    fig, axes = plt.subplots(int(np.ceil(len(label_to_class) / 2)), 2, figsize=(10, 20))
    axes = axes.flatten()
    for i, conf_matrix in enumerate(multilabel_confusion_matrix(testY, y_pred)):
        tn, fp, fn, tp = conf_matrix.ravel()
        f1 = 2 * tp / (2 * tp + fp + fn + sys.float_info.epsilon)
        recall = tp / (tp + fn + sys.float_info.epsilon)
        precision = tp / (tp + fp + sys.float_info.epsilon)
        plot_confusion_matrix(
            np.array([[tp, fn], [fp, tn]]),
            classes=['+', '-'],
            title=f'Label: {label_to_class[i]}\nf1={f1:.5f}\nrecall={recall:.5f}\nprecision={precision:.5f}',
            ax=axes[i]
        )
        plt.tight_layout()
    if save_plot:
        plt.savefig('confusion_matrices_smallvgg.png')

#wandb.log({'Confusion matrix':plot_ml_confusion_matrix(testY, y_pred,label_to_class,save_plot=True)})

def plot_roc_curve(testY, y_prob, label_to_class):
    lw = 2
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(label_to_class):
        fpr[i], tpr[i], _ = roc_curve(testY[:, i], y_prob[:, i])
        roc_auc[i] = auc(fpr[i], tpr[i])
    
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(testY.ravel(), y_prob.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    
    all_fpr = np.unique(np.concatenate([fpr[i] for i in range(label_to_class)]))
    
    # Then interpolate all ROC curves at this points
    mean_tpr = np.zeros_like(all_fpr)
    for i in range(label_to_class):
        mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
    mean_tpr /= label_to_class

    fpr["macro"] = all_fpr
    tpr["macro"] = mean_tpr
    roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])
    # Plot all ROC curves
    plt.figure()
    plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

   # plt.plot(fpr["macro"], tpr["macro"],
         #label='macro-average ROC curve (area = {0:0.2f})'
          #     ''.format(roc_auc["macro"]),
         #color='navy', linestyle=':', linewidth=4)

    colors = cycle(['yellowgreen', 'darkorange', 'cornflowerblue', 'green', 'red', 'brown','gold'])
    labels = mlb.classes_
    for (i, color, label) in zip(range(label_to_class), colors, labels):
        plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(label, roc_auc[i]))
        
    plt.plot([0, 1], [0, 1], 'k--', lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    #plt.title('Roc Curve for each class')
    plt.legend(loc="lower right")
    plt.grid()
    plt.savefig('roc_smallvgg_2k.png')
    return plt

wandb.log({'ROC':plot_roc_curve(testY, y_prob,label_to_class)})
#wandb.Image({'ROC':plot_roc_curve(testY, y_pred,label_to_class)})
