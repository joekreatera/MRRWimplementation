import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from  mlmrrw.MLMRRWPredictor import MLMRRWPredictor
from mlmrrw.DownloadHelper import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import average_precision_score, accuracy_score, hamming_loss, classification_report, multilabel_confusion_matrix
from sklearn.metrics import label_ranking_average_precision_score
from random import random
from sklearn.metrics import roc_auc_score
from sklearn import preprocessing

from sklearn.preprocessing import StandardScaler, QuantileTransformer

from mlmrrw.utils import generate_compatibility_matrix_counting_0s, generate_cosine_distance_based_compatibility,generate_compatibility_matrix, transform_multiclass_to_multilabel, get_features, save_report
import sys


ds_name = "emotions"

ds_configs = {
"emotions":6, # multilabel
}



# getFromOpenML will convert automatically the classes found to a mutually exclusive multilabel
dataset = getFromOpenML(ds_name,version="active",ospath='datasets/', download=False, save=False)


# multilabel_dataset = transform_multiclass_to_multilabel(dataset, "label_0") # will expand label_0 to the unique values , mutually exclusive as labels
train_set,test_set = train_test_split(dataset, test_size=.30, random_state=80)

# train_set = train_set.iloc[0:30]
# 50% unlabeled
labeled_instances, unlabeled_instances =  train_test_split(train_set, test_size=.5, random_state=101) # simulate unlabeled instances


# columns list 
label_columns = [f"label_{i}" for i in range(0,ds_configs[ds_name])]  # for iris (3) ,for yeast(10) for ecoli(8), satimage(6)
instance_columns = get_features(train_set, label_columns)
#print(train_set)



X = train_set[instance_columns]
scaler = None
y = train_set[label_columns]


# change them according to original paper
parameters = {
'gamma_A':0.001 ,
'gamma_I':19 ,
'XI_v1': 0.9, 
'XI_v2': 0.1,
'gamma':30,
'eps': 0.7}
# XI is the coefficient for supervised/unsupervised trust
predictor = MLMRRWPredictor(
                            unlabeledIndex=unlabeled_instances.index,
                            tag=ds_name,
                            hyper_params_dict = parameters,
                            )
                                                        
predictor.fit(X,y)


y_true = test_set[label_columns].to_numpy()
x_test = test_set[instance_columns]

predictions, probabilities = predictor.predict_with_proba(x_test) #y_true = y_true



print(predictions)
print(probabilities)

o = save_report('./output', ds_name, y_true, predictions, probabilities, do_output=True, parameters=parameters)
print(o)