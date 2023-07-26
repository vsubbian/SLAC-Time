#Import modules
from preprocess import *
from util import *
import clustering

# Numpy for numerical computation
import numpy as np

# TensorFlow for deep learning models
import tensorflow as tf

# Pandas for data manipulation
import pandas as pd

# GC for garbage collection to manage memory
import gc

# TQDM for progress bar
from tqdm import tqdm

# Different components from TensorFlow for creating and training the deep learning models
from tensorflow.keras.layers import Layer, Input, Dense, Embedding, Add, Lambda, Concatenate
from tensorflow.keras import Model
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.nn import dropout as nn_dropout
from tensorflow.python.keras.utils import tf_utils
from tensorflow.python.ops import array_ops
from tensorflow import nn

# Iterative Imputer for imputing missing data
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer


# Define model architecture
class CVE(Layer):
    def __init__(self, hid_units, output_dim):
        self.hid_units = hid_units
        self.output_dim = output_dim
        super(CVE, self).__init__()
        
    def build(self, input_shape): 
        self.W1 = self.add_weight(name='CVE_W1',
                            shape=(1, self.hid_units),
                            initializer='glorot_uniform',
                            trainable=True)
        self.b1 = self.add_weight(name='CVE_b1',
                            shape=(self.hid_units,),
                            initializer='zeros',
                            trainable=True)
        self.W2 = self.add_weight(name='CVE_W2',
                            shape=(self.hid_units, self.output_dim),
                            initializer='glorot_uniform',
                            trainable=True)
        super(CVE, self).build(input_shape)
        
    def call(self, x):
        x = K.expand_dims(x, axis=-1)
        x = K.dot(K.tanh(K.bias_add(K.dot(x, self.W1), self.b1)), self.W2)
        return x
        
    def compute_output_shape(self, input_shape):
        return input_shape + (self.output_dim,)
    
    
class Attention(Layer):
    
    def __init__(self, hid_dim):
        self.hid_dim = hid_dim
        super(Attention, self).__init__()

    def build(self, input_shape):
        d = input_shape.as_list()[-1]
        self.W = self.add_weight(shape=(d, self.hid_dim), name='Att_W',
                                 initializer='glorot_uniform',
                                 trainable=True)
        self.b = self.add_weight(shape=(self.hid_dim,), name='Att_b',
                                 initializer='zeros',
                                 trainable=True)
        self.u = self.add_weight(shape=(self.hid_dim,1), name='Att_u',
                                 initializer='glorot_uniform',
                                 trainable=True)
        super(Attention, self).build(input_shape)
        
    def call(self, x, mask, mask_value=-1e30):
        attn_weights = K.dot(K.tanh(K.bias_add(K.dot(x,self.W), self.b)), self.u)
        mask = K.expand_dims(mask, axis=-1)
        attn_weights = mask*attn_weights + (1-mask)*mask_value
        attn_weights = K.softmax(attn_weights, axis=-2)
        return attn_weights
        
    def compute_output_shape(self, input_shape):
        return input_shape[:-1] + (1,)
    
class Transformer(Layer):
    
    def __init__(self, N=2, h=8, dk=None, dv=None, dff=None, dropout=0):
        self.N, self.h, self.dk, self.dv, self.dff, self.dropout = N, h, dk, dv, dff, dropout
        self.epsilon = K.epsilon() * K.epsilon()
        super(Transformer, self).__init__()

    def build(self, input_shape):
        d = input_shape.as_list()[-1]
        if self.dk==None:
            self.dk = d//self.h
        if self.dv==None:
            self.dv = d//self.h
        if self.dff==None:
            self.dff = 2*d
        self.Wq = self.add_weight(shape=(self.N, self.h, d, self.dk), name='Wq',
                                 initializer='glorot_uniform', trainable=True)
        self.Wk = self.add_weight(shape=(self.N, self.h, d, self.dk), name='Wk',
                                 initializer='glorot_uniform', trainable=True)
        self.Wv = self.add_weight(shape=(self.N, self.h, d, self.dv), name='Wv',
                                 initializer='glorot_uniform', trainable=True)
        self.Wo = self.add_weight(shape=(self.N, self.dv*self.h, d), name='Wo',
                                 initializer='glorot_uniform', trainable=True)
        self.W1 = self.add_weight(shape=(self.N, d, self.dff), name='W1',
                                 initializer='glorot_uniform', trainable=True)
        self.b1 = self.add_weight(shape=(self.N, self.dff), name='b1',
                                 initializer='zeros', trainable=True)
        self.W2 = self.add_weight(shape=(self.N, self.dff, d), name='W2',
                                 initializer='glorot_uniform', trainable=True)
        self.b2 = self.add_weight(shape=(self.N, d), name='b2',
                                 initializer='zeros', trainable=True)
        self.gamma = self.add_weight(shape=(2*self.N,), name='gamma',
                                 initializer='ones', trainable=True)
        self.beta = self.add_weight(shape=(2*self.N,), name='beta',
                                 initializer='zeros', trainable=True)
        super(Transformer, self).build(input_shape)
        
    def call(self, x, mask, mask_value=-1e-30):
        mask = K.expand_dims(mask, axis=-2)
        for i in range(self.N):
            # MHA
            mha_ops = []
            for j in range(self.h):
                q = K.dot(x, self.Wq[i,j,:,:])
                k = K.permute_dimensions(K.dot(x, self.Wk[i,j,:,:]), (0,2,1))
                v = K.dot(x, self.Wv[i,j,:,:])
                A = K.batch_dot(q,k)
                # Mask unobserved steps.
                A = mask*A + (1-mask)*mask_value
                # Mask for attention dropout.
                def dropped_A():
                    dp_mask = K.cast((K.random_uniform(shape=array_ops.shape(A))>=self.dropout), K.floatx())
                    return A*dp_mask + (1-dp_mask)*mask_value
                A = tf_utils.smart_cond(K.learning_phase(), dropped_A, lambda: array_ops.identity(A))
                A = K.softmax(A, axis=-1)
                mha_ops.append(K.batch_dot(A,v))
            conc = K.concatenate(mha_ops, axis=-1)
            proj = K.dot(conc, self.Wo[i,:,:])
            # Dropout.
            proj = tf_utils.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(proj, rate=self.dropout)),                                       lambda: array_ops.identity(proj))
            # Add & LN
            x = x+proj
            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x*self.gamma[2*i] + self.beta[2*i]
            # FFN
            ffn_op = K.bias_add(K.dot(K.relu(K.bias_add(K.dot(x, self.W1[i,:,:]), self.b1[i,:])), 
                           self.W2[i,:,:]), self.b2[i,:,])
            # Dropout.
            ffn_op = tf_utils.smart_cond(K.learning_phase(), lambda: array_ops.identity(nn.dropout(ffn_op, rate=self.dropout)),                                       lambda: array_ops.identity(ffn_op))
            # Add & LN
            x = x+ffn_op
            mean = K.mean(x, axis=-1, keepdims=True)
            variance = K.mean(K.square(x - mean), axis=-1, keepdims=True)
            std = K.sqrt(variance + self.epsilon)
            x = (x - mean) / std
            x = x*self.gamma[2*i+1] + self.beta[2*i+1]            
        return x
        
    def compute_output_shape(self, input_shape):
        return input_shape

def build_strats(max_len, V, d, N, he, dropout, forecast=False):
    varis = Input(shape=(max_len,))
    values = Input(shape=(max_len,))
    times = Input(shape=(max_len,))
    varis_emb = Embedding(V+1, d)(varis)
    cve_units = int(np.sqrt(d))
    values_emb = CVE(cve_units, d)(values)
    times_emb = CVE(cve_units, d)(times)
    comb_emb = Add()([varis_emb, values_emb, times_emb]) # b, L, d
    mask = Lambda(lambda x:K.clip(x,0,1))(varis) # b, L
    cont_emb = Transformer(N, he, dk=None, dv=None, dff=None, dropout=dropout)(comb_emb, mask=mask)
    attn_weights = Attention(2*d)(cont_emb, mask=mask)
    fused_emb = Lambda(lambda x:K.sum(x[0]*x[1], axis=-2))([cont_emb, attn_weights])
    fore_op = Dense(V)(fused_emb)
    op = Dense(3, activation='softmax')(fused_emb)
    model = Model([times, values, varis], op)
    if forecast:
        fore_model = Model([times, values, varis], fore_op)
        return [model, fore_model]
    return model



#Initial loading of the physiologic data
pds = physiologic_data_sampling(time_stamp=10, sample_len=30*60, sample_stride=30*60)
(value, mask, output), (pat_to_ind, var_to_ind) = pds.sampling()
# input_vars = list(var_to_ind.keys())


n=value.shape[0]*value.shape[1]*value.shape[2]
df1=np.zeros((n,4))
counter = 0
for s in range(value.shape[0]):
    for t in range(value.shape[1]):
        for v in range(value.shape[2]):
            df1[counter,:] = np.array([s, t, v,value[s,t,v]])
            counter = counter + 1

#make a DataFrame from the value numpy array
data=pd.DataFrame(df1, columns=["ts_ind","time",'vind',"value"])
data=data.dropna()


# Normalize data 
means_stds = data.groupby('vind').agg({'value':['mean', 'std']})
means_stds.columns = [col[1] for col in means_stds.columns]
means_stds.loc[means_stds['std']==0, 'std'] = 1
data = data.merge(means_stds.reset_index(), on='vind', how='left')
data['value'] = (data['value']-data['mean'])/data['std']


# Training and validation indices
train_valid_ind = np.array(data.ts_ind.unique()).astype(int)


# Get the number of variables.
varis = sorted(list(set(data.vind)))
V = len(varis)

#Number of samples
N = max(data['ts_ind'])+1


# Generate split
# train_valid_ind = np.arange(0, max(data['ts_ind'])+1)
np.random.seed(123)
np.random.shuffle(train_valid_ind)
bp = int(0.8*len(train_valid_ind))
train_ind = train_valid_ind[:bp]
valid_ind = train_valid_ind[bp:]


data = data[['ts_ind', 'vind', 'time', 'value']].sort_values(by=['ts_ind', 'vind', 'time'])

# Find max_len.
input_max_len = data.groupby('ts_ind').size().max()
print ('input_max_len', input_max_len)


#Load proxy task dataset into matrices
pred_window = 5 #time steps 
obs_windows = [20, 40, 80, 160,175]

fore_times_ip = []
fore_values_ip = []
fore_varis_ip = []
fore_op = []
fore_inds = []

def f(x):
    mask = [0 for i in range(V)]
    values = [0 for i in range(V)]
    for vv in x:
        v = int(vv[0])-1
        mask[v] = 1
        values[v] = vv[1]
    return values+mask

def pad(x):
    return x+[0]*(input_max_len-len(x))

for w in tqdm(obs_windows):
    pred_data = data.loc[(data.time>=w)&(data.time<=w+pred_window)]
    pred_data = pred_data.groupby(['ts_ind', 'vind']).agg({'value':'first'}).reset_index()
    pred_data['vind_value'] = pred_data[['vind', 'value']].values.tolist()
    pred_data = pred_data.groupby('ts_ind').agg({'vind_value':list}).reset_index()
    pred_data['vind_value'] = pred_data['vind_value'].apply(f)    
    obs_data = data.loc[data.time<w]
    obs_data = obs_data.loc[obs_data.ts_ind.isin(pred_data.ts_ind)]
    obs_data = obs_data.groupby('ts_ind').agg({'vind':list, 'time':list, 'value':list}).reset_index()
    obs_data = obs_data.merge(pred_data, on='ts_ind')
    for col in ['vind', 'time', 'value']:
        obs_data[col] = obs_data[col].apply(pad)
    fore_op.append(np.array(list(obs_data.vind_value)))
    fore_inds.append(np.array(list(obs_data.ts_ind)))
    fore_times_ip.append(np.array(list(obs_data.time)))
    fore_values_ip.append(np.array(list(obs_data.value)))
    fore_varis_ip.append(np.array(list(obs_data.vind)))

fore_times_ip = np.concatenate(fore_times_ip, axis=0).astype("int16")
fore_values_ip = np.concatenate(fore_values_ip, axis=0).astype("float16")
fore_varis_ip = np.concatenate(fore_varis_ip, axis=0).astype("int16")
fore_op = np.concatenate(fore_op, axis=0).astype("float16")
fore_inds = np.concatenate(fore_inds, axis=0).astype("int32")

# Generate 3 sets of inputs and outputs.
train_ind = np.argwhere(np.in1d(fore_inds, train_ind)).flatten()
valid_ind = np.argwhere(np.in1d(fore_inds, valid_ind)).flatten()
fore_train_ip = [ip[train_ind] for ip in [fore_times_ip, fore_values_ip, fore_varis_ip]]
fore_valid_ip = [ip[valid_ind] for ip in [fore_times_ip, fore_values_ip, fore_varis_ip]]
fore_train_op = fore_op[train_ind]
fore_valid_op = fore_op[valid_ind]

#clear memory
del fore_times_ip, fore_values_ip, fore_varis_ip, fore_op
gc.collect()            
            
#Load target task dataset into matrices
# Generate split
np.random.shuffle(train_valid_ind)
bp = int(0.8*len(train_valid_ind))
target_train_ind = train_valid_ind[:bp]
target_valid_ind = train_valid_ind[bp:]


data = data[['ts_ind', 'vind', 'time', 'value']].sort_values(by=['ts_ind', 'vind', 'time'])  

target_times_ip = []
target_values_ip = []
target_varis_ip = []
target_inds = []

    
data = data.groupby('ts_ind').agg({'vind':list, 'time':list, 'value':list}).reset_index()

for col in ['vind', 'time', 'value']:
    data[col] = data[col].apply(pad)
print(data.ts_ind)


target_inds.append(np.array(list(data.ts_ind)))
target_times_ip.append(np.array(list(data.time)))
target_values_ip.append(np.array(list(data.value)))
target_varis_ip.append(np.array(list(data.vind))) 
    
target_times_ip = np.concatenate(target_times_ip, axis=0).astype("int16")
target_values_ip = np.concatenate(target_values_ip, axis=0).astype("float16")
target_varis_ip = np.concatenate(target_varis_ip, axis=0).astype("int16")
target_inds = np.concatenate(target_inds, axis=0).astype("int32")
print(target_inds)
len(target_inds)


dataset = [ip[target_inds] for ip in [target_times_ip, target_values_ip, target_varis_ip]]

target_times_ip_tensor=tf.constant(dataset[0])
target_values_ip_tensor=tf.constant(dataset[1])
target_varis_ip_tensor=tf.constant(dataset[2])
dataset=[target_times_ip_tensor, target_values_ip_tensor, target_varis_ip_tensor]

    
target_train_ip = [ip[target_train_ind] for ip in [target_times_ip, target_values_ip, target_varis_ip]]
target_valid_ip = [ip[target_valid_ind] for ip in [target_times_ip, target_values_ip, target_varis_ip]]


target_train_times_ip_tensor=tf.constant(target_train_ip[0])
target_train_values_ip_tensor=tf.constant(target_train_ip[1])
target_train_varis_ip_tensor=tf.constant(target_train_ip[2])
target_train_ip_tensor=[target_train_times_ip_tensor,target_train_values_ip_tensor,target_train_varis_ip_tensor]


target_valid_times_ip_tensor=tf.constant(target_valid_ip[0])
target_valid_values_ip_tensor=tf.constant(target_valid_ip[1])
target_valid_varis_ip_tensor=tf.constant(target_valid_ip[2])
target_valid_ip_tensor=[target_valid_times_ip_tensor, target_valid_values_ip_tensor, target_valid_varis_ip_tensor]


#clear memory
del target_times_ip, target_values_ip, target_varis_ip, target_train_times_ip_tensor, target_train_values_ip_tensor, target_train_varis_ip_tensor, target_valid_times_ip_tensor, target_valid_values_ip_tensor,target_valid_varis_ip_tensor
gc.collect()


def forecast_loss(y_true, y_pred):
    return K.sum(y_true[:,V:]*(y_true[:,:V]-y_pred)**2, axis=-1)


def get_min_loss(weight):
    def min_loss(y_true, y_pred):
        return weight*y_pred
    return min_loss


# Pretrain on forecasting (Proxy task)
lr, batch_size, samples_per_epoch, patience = 0.0005, 8, len(fore_train_op), 10
d, N, he, dropout = 100, 2, 4, 0.2
model, fore_model =  build_strats(input_max_len, V, d, N, he, dropout, forecast=True) 
print (fore_model.summary())
fore_model.compile(loss=forecast_loss, optimizer=Adam(lr))

# Pretrain fore_model.
best_val_loss = np.inf
N_fore = len(fore_train_op)
print(N_fore)


for e in range(200):
    tf.keras.backend.clear_session()
    e_indices = np.random.choice(range(N_fore), size=samples_per_epoch, replace=False)
    e_loss = 0
    for start in tqdm(range(0, len(e_indices), batch_size)):
        tf.keras.backend.clear_session()
        ind = e_indices[start:start+batch_size]
        inp=[ip[ind] for ip in fore_train_ip]
        out=fore_train_op[ind]
        fore_times_ip_tensor=tf.constant(inp[0])
        fore_values_ip_tensor=tf.constant(inp[1])
        fore_varis_ip_tensor=tf.constant(inp[2])
        fore_train_ip_tensor=[fore_times_ip_tensor,fore_values_ip_tensor,fore_varis_ip_tensor]
        fore_train_op_tensor=tf.constant(out)
        e_loss += fore_model.train_on_batch(fore_train_ip_tensor, fore_train_op_tensor)    
    fore_v_times_ip_tensor=tf.constant(fore_valid_ip[0])
    fore_v_values_ip_tensor=tf.constant(fore_valid_ip[1])
    fore_v_varis_ip_tensor=tf.constant(fore_valid_ip[2])
    fore_valid_ip_tensor= [fore_v_times_ip_tensor,fore_v_values_ip_tensor,fore_v_varis_ip_tensor]
    fore_valid_op_tensor=tf.constant(fore_valid_op)
    val_loss = fore_model.evaluate(fore_valid_ip_tensor, fore_valid_op_tensor, batch_size=batch_size, verbose=1)
    print ('Epoch', e, 'loss', e_loss*batch_size/samples_per_epoch, 'val loss', val_loss)
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        fore_model.save_weights('physiologic_forecast_weights.h5')
        best_epoch = e
    if (e-best_epoch)>patience:
        break


#clear memory
del fore_model, fore_times_ip_tensor,fore_values_ip_tensor,fore_varis_ip_tensor,fore_train_ip_tensor,fore_train_op_tensor, fore_v_times_ip_tensor, fore_v_values_ip_tensor, fore_v_varis_ip_tensor, fore_valid_ip_tensor,fore_valid_op_tensor
gc.collect()


# Transfer the weights into the target model
# Build and compile model.
d, N, he, dropout = 100, 2, 4, 0.2
model, fore_model =  build_strats(input_max_len, V, d, N, he, dropout, forecast=True)
model.compile(loss='sparse_categorical_crossentropy', optimizer=Adam(lr))
fore_model.compile(loss=forecast_loss, optimizer=Adam(lr))

# Load pretrained weights here.
fore_model.load_weights('physiologic_forecast_weights.h5')

# clustering algorithm to use
deepcluster = clustering.__dict__['Kmeans'](3)


# training STraTS
cluster_assignments=[]
for iteration in range(0,500):
    tf.keras.backend.clear_session()
    model_features = Model(model.input, model.layers[-2].output)
    features = model_features.predict(dataset)
    clustering_loss = deepcluster.cluster(np.array(features))
    labeled_dataset = clustering.cluster_assign(deepcluster.samples_lists, dataset)
    
    # Building new inputs and outputs
    target_input=dataset
    target_output = np.array([labeled_dataset.ptns[i][1] for i in range(len(labeled_dataset.ptns))])
    target_train_op_tensor = tf.constant(target_output[target_train_ind])
    target_valid_op_tensor = tf.constant(target_output[target_valid_ind])
   
    monitor = EarlyStopping(monitor='val_loss', min_delta=1e-3, patience=5, 
        verbose=1, mode='auto', restore_best_weights=True)
    model.fit(target_train_ip_tensor, target_train_op_tensor, batch_size=8, validation_data=(target_valid_ip_tensor,target_valid_op_tensor),
        callbacks=[monitor],verbose=2,epochs=200)
    
    model.save_weights('physiologic_target_weights.h5')
    
    #clear memory
    del features, clustering_loss, labeled_dataset, target_input, target_output, target_train_op_tensor, target_valid_op_tensor
    gc.collect()

    # save cluster assignments
    cluster_assignments.append(clustering.arrange_clustering(deepcluster.samples_lists))
    with open('cluster_assignments.npy', 'wb') as f:
        np.save(f, cluster_assignments)

model.save_weights('physiologic_target_weights_architecture.h5')
