from tensorflow import keras
from tensorflow.python.keras.models import load_model
import numpy as np
import itertools
import argparse
import pickle
import os
import matplotlib.pyplot as plt
import matplotlib
from matplotlib import image, pyplot as plt
from random import sample, randint
import wandb
from preprocessor import preprocessor
from itertools import cycle
from scipy import interp
from sklearn.metrics import classification_report
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score
wandb.init(project="master_thesis", name="classify_models")

# construct the argument parse and parse the arguments
ap = argparse.ArgumentParser()
ap.add_argument("-m", "--model", required=True,
        help="path to trained model model")
ap.add_argument("-l", "--labelbin", required=True,
        help="path to label binarizer")
ap.add_argument("-m1", "--model1", required=True,
        help="path to trained model model")
ap.add_argument("-m2", "--model2", required=True,
        help="path to trained model model")
ap.add_argument("-m3", "--model3", required=True,
        help="path to trained model model")
ap.add_argument("-m4", "--model4", required=True,
        help="path to trained model model")

args = vars(ap.parse_args())

#load data
f = open("/var/scratch/nsc400/hera_data/HERA_02-05-2020.pkl","rb")
dataset = pickle.load(f,encoding='latin1')
data = dataset[0]
labels = dataset[2]

label = [l.split('-') for l in labels]
labels = np.array(label,dtype=object)
as_list = [list(i) for i in labels]

#prepocess data
#resize
p = preprocessor(data)
p.interp(32,128)
p.get_magnitude_and_phase()
p.median_threshold()
p.minmax(per_baseline=True,feature_range=(0,1))
processed_data = p.get_processed_cube()

#prepocess data
#resize
p1 = preprocessor(data)
p1.interp(32,128)
p1.get_magnitude()
processed_data2 = p1.get_processed_cube()[...,0:1]

# load the trained convolutional neural network and the multi-label
# binarizer
print("[INFO] loading network...")
model = load_model(args["model"])
model1 = load_model(args["model1"])
model2 = load_model(args["model2"])
model3 = load_model(args["model3"])
model4 = load_model(args["model4"])

mlb = pickle.loads(open(args["labelbin"], "rb").read())
ohe = mlb.fit_transform(as_list)

label_to_class = len(mlb.classes_)

output = model.predict(processed_data)
output = np.array([[1 if i > 0.03  else 0 for i in j] for j in output])

#print classification report
print(classification_report(ohe, output))

def plot_analysis(model,data,labels):

    output  = model.predict(data)
    output = np.array([[1 if i > 0.03  else 0 for i in j] for j in output])
    #titles= mlb.inverse_transform(labels)
    titles = as_list
    it = 1
    if isinstance(data,list):
        it = 2
        shape = output[0].shape[0]
    else:
        shape = output.shape[0]

    for i in range(it):
        fig,axs = plt.subplots(3,2,figsize=(20,10))
        fig.subplots_adjust(hspace=0.5)

        if isinstance(data,list):
            inp = data[i]
            outp = output[i]
        else:
            inp = data
            outp = output

        for i in range(3):
            r = randint(0,shape-1)
            #o = output[sample]

            axs[i,0].set_title(f' {titles[r]}',rotation=10,fontsize=10 )
            axs[i,0].imshow(inp[r,...,0])

            axs[i,1].bar(range(len(mlb.classes_)), outp[r])
            axs[i,1].set_xticks(range(len(mlb.classes_)))
            axs[i,1].set_xticklabels(["{}. {}".format(i+1, label) for (i, label) in enumerate(mlb.classes_)],rotation=40)

        return plt

wandb.log({'Analysis':plot_analysis(model,processed_data,as_list)})

output1 = model1.predict(processed_data2, verbose=1)
output2 = model2.predict(processed_data2, verbose=1)
output3 = model3.predict(processed_data2, verbose=1)
output4 = model4.predict(processed_data2, verbose=1)
binaryp1 = (output1 >0.03).astype(np.uint8)
binaryp2 = (output2 >0.03).astype(np.uint8)
binaryp3 = (output3 > 0.5).astype(np.uint8)
binaryp4 = (output4 >0.5).astype(np.uint8)

#print('Classification report 1:\n', classification_report(mask1.flatten(), binaryp1.flatten()))
#print('Classification report 2:\n', classification_report(mask2.flatten(), binaryp2.flatten()))
#print('Classification report 3:\n', classification_report(mask3.flatten(), binaryp3.flatten()))
#print('Classification report 4:\n', classification_report(mask4.flatten(), binaryp4.flatten()))

def plot_io(model,model2,data,data2,output,output2,binaryp1,binaryp2,binaryp3,binaryp4,as_list):
    output  = output
    output2 = output2
    binaryp1 = binaryp1
    binaryp2 = binaryp2
    binaryp3 = binaryp3
    binaryp4 = binaryp4
    titles = as_list
    #print(model.evaluate(data, mask, verbose=1))
    it = 1
    if isinstance(data,list):
        it = 2
        shape = output[0].shape[0]
    else:
        shape = output.shape[0]

    for i in range(it):
        fig,axs = plt.subplots(6,2,figsize=(10,10))
        fig.subplots_adjust(hspace=1)

        if isinstance(data,list):
            inp = data[i]
            inp2 = data2[i]
            outp = output[i]
            outp2 = output2[i]
            bp1 = binaryp1[i]
            bp2 = binaryp2[i]
            bp3 = binaryp3[i]
            bp4 = binaryp4[i]
        else:
            inp = data
            inp2 = data2
            outp = output
            bp1 = binaryp1
            bp2 = binaryp2
            bp3 = binaryp3
            bp4 = binaryp4

        for i in range(2):
            r = randint(0,shape-1)
            
            axs[0,i].set_title(f' {titles[r]}',fontsize=10 )
            axs[0,i].imshow(inp[r,...,0])

            axs[1,i].bar(range(len(mlb.classes_)), outp[r])
            axs[1,i].set_xticks(range(len(mlb.classes_)))
            axs[1,i].set_xticklabels(["{}.{}".format(i+1, label) for (i, label) in enumerate(mlb.classes_)],rotation=22,fontsize=7)

            #axs[i,2].imshow(inp2[r,...,0]);
            #if has_mask:
                #axs[0,j].contour(msk[r,...,0].squeeze(), levels=[0.1])
            #axs[i,2].set_title(f' {labels[r]}',fontsize=10)

            axs[2,i].imshow(bp1[r,...,0].squeeze(), vmin=0, vmax=1);
            axs[2,i].title.set_text('Mask Predicted for point source {}'.format(r))

            axs[3,i].imshow(bp2[r,...,0].squeeze(), vmin=0, vmax=1);
            axs[3,i].title.set_text('Mask Predicted for rfi impulse {}'.format(r))

            axs[4,i].imshow(bp3[r,...,0].squeeze(), vmin=0, vmax=1);
            axs[4,i].title.set_text('Mask Predicted for rfi stations {}'.format(r))

            axs[5,i].imshow(bp4[r,...,0].squeeze(), vmin=0, vmax=1);
            axs[5,i].title.set_text('Mask Predicted for rfi dtv {}'.format(r))

        return plt

wandb.log({'Analysis1':plot_io(model,model1,processed_data,processed_data2,output,output1,binaryp1,binaryp2,binaryp3,binaryp4,as_list)})
#wandb.log({'Analysis2':plot_io(model,model2,processed_data,processed_data2,output,output2,binaryp2,as_list)})
#wandb.log({'Analysis3':plot_io(model,model3,processed_data,processed_data2,output,output3,binaryp3,as_list)})
#wandb.log({'Analysis4':plot_io(model,model4,processed_data,processed_data2,output,output4,binaryp4,as_list)})
