import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import parser
import model

#set dataset directory
parent_dir = 'UrbanSound8k/audio'
#set training and testing directories
#tr_sub_dirs = ["fold1","fold2","fold3","fold4","fold5","fold6","fold7","fold8","fold9"]
tr_sub_dirs = ['fold2','fold3','fold5']
ts_sub_dirs = ["fold10"]
#get features and labels
tr_features, tr_labels = parser.parse_audio_files(parent_dir,tr_sub_dirs)
ts_features, ts_labels = parser.parse_audio_files(parent_dir,ts_sub_dirs)
#encode labels
tr_labels = parser.one_hot_encode(tr_labels)
ts_labels = parser.one_hot_encode(ts_labels)
#run model
p,r,f,s = model.create(tr_features,tr_labels,ts_features,ts_labels)
