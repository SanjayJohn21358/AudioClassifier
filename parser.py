import glob
import os
import librosa
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram

def extract_feature(file_name):

    """
    extract mel-frequency cepstral coefficients, chromagraph, 
    spectral contrast, tonal centroid features from file

    :file_name: input, file name, str

    :mfccs: output, mel-frequency cepstral coefficients, list float
    :chroma: output, chromagraph, list float
    :mel: output, mel spectrogram, list float
    :constrast: output, spectral constrast, list float
    :tonnetz: output, tonal centroid features, list float

    """

    X, sample_rate = librosa.load(file_name)
    stft = np.abs(librosa.stft(X))
    mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T,axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T,axis=0)
    mel = np.mean(librosa.feature.melspectrogram(X, sr=sample_rate).T,axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(S=stft, sr=sample_rate).T,axis=0)
    tonnetz = np.mean(librosa.feature.tonnetz(y=librosa.effects.harmonic(X),sr=sample_rate).T,axis=0)
    return mfccs,chroma,mel,contrast,tonnetz

def parse_audio_files(parent_dir,sub_dirs,file_ext="*.wav"):

    """
    parse all audio and extract features, attach corresonding labels
    :parent_dir: input, parent directory, str
    :sub_dirs: input, sub directory, str
    :file_ext: input, file extension, str

    :features: output, features from extract_feature, np array
    :labels: output, labels of audio files, np array

    """

    features, labels = np.empty((0,193)), np.empty(0)
    for label, sub_dir in enumerate(sub_dirs):
        for fn in glob.glob(os.path.join(parent_dir, sub_dir, file_ext)):
            try:
              mfccs, chroma, mel, contrast,tonnetz = extract_feature(fn)
            except Exception as e:
              print("Error encountered while parsing file: ", fn)
              continue
            ext_features = np.hstack([mfccs,chroma,mel,contrast,tonnetz])
            features = np.vstack([features,ext_features]) 
            labels = np.append(labels, fn.split('/')[3].split('-')[1]) #labels are present in filename
    return np.array(features), np.array(labels, dtype = np.int)

def one_hot_encode(labels):
    """
    returns one-hot encoding of labels for use in NN
    """

    n_labels = len(labels)
    n_unique_labels = len(np.unique(labels))
    one_hot_encode = np.zeros((n_labels,n_unique_labels))
    one_hot_encode[np.arange(n_labels), labels] = 1
    return one_hot_encode

