import numpy as np
from tensorflow.keras import backend as K
import tensorflow as tf


DATA_PATH = 'data/'
PHYSIO_DATA_PATH = 'data/physio data/'

def inv_list(l):
    d = {}
    for i in range(len(l)):
        d[l[i]] = i
    return d

def extr_org_sample(sample_id, sample_to_patients, df_3d, var_to_ind):
    p_id = sample_to_patients.iloc[sample_id, 0]
    start = sample_to_patients.iloc[sample_id, 1]
    stop = sample_to_patients.iloc[sample_id, 2]
    seq = np.zeros((stop-start, len(var_to_ind)))
    for i in list(var_to_ind.keys()):
        temp_seq = df_3d[(df_3d.patient_id==p_id) & (df_3d.variable==i)].value_mask.values[0][0,:]
        temp_seq = temp_seq[start:stop]
        seq[:, var_to_ind[i]] = temp_seq.copy()
    return seq

class TimeSeriesScaler():
    def __init__(self):
        self.mean = None
        self.std = None
    
    def fit(self, x):
        assert len(x.shape) == 3
        self.mean = x.mean(axis=(0,1))
        self.std = x.std(axis=(0,1))
        
    def fit_transform(self, x):
        assert len(x.shape) == 3
        # Saving the mean and std only for the first time
        if self.mean is None:
            self.mean = x.mean(axis=(0,1))
            self.std = x.std(axis=(0,1))
        
        return (x - self.mean.reshape((1, 1, x.shape[2])))/self.std.reshape((1, 1, x.shape[2]))
    
    def inverse_transfrom(self, y):
        assert len(y.shape) == 3
        return y * self.std.reshape((1, 1, y.shape[2])) + self.mean.reshape((1, 1, y.shape[2]))

def isfloat(string):
    try:
        float(string)
        return True
    except ValueError:
        return False