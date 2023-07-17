import os
import gc
import h5py
import math
import pickle
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers, regularizers
from sklearn.model_selection import train_test_split
import keras.backend as K

# Tensorflow compiler flags
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Tensorflow GPU usage
print("\n>>> Num GPUs Available: ", len(tf.config.list_physical_devices('GPU')), '\n')

### Choosing network model ###
net_dict = {1:'trans', 2:'rot'}
net = int(input('Wich network to train?\n  [1] Translation\n  [2] Rotation\nChoose: '))

### Choosing vessel
print('\nAvailable vessels: VG, VM and VP')
vessel = input('Type the desired option: ')

path = f'path/to/train/files'

### Loading files ###
array_file = open(path + 'canon_points.npy', 'rb')
points = np.load(array_file, allow_pickle=True)
points = points.astype(np.float32)

array_file = open(path + 'T.npy', 'rb')
targets = np.load(array_file, allow_pickle=True)

if net == 1:
  # trans targets
  targets = [[i[1,3], i[2,3]] for i in targets]
else:
  #rot targets
  targets = [np.concatenate((np.array(T[:3,:3][:,0]), np.array(T[:3,:3][:,1]))) for T in targets]

print('Files loaded!')


### Splitting data ###
train_points, valid_points, train_targets, valid_targets = train_test_split(points, targets, train_size=0.9, shuffle=True)

def augment(points, label):
    # jitter points
    points += tf.random.uniform(points.shape, -0.00005, 0.00005, dtype=tf.float32)
    # shuffle points
    points = tf.random.shuffle(points)
    return points, label


train_dataset = tf.data.Dataset.from_tensor_slices((train_points, train_targets))
valid_dataset = tf.data.Dataset.from_tensor_slices((valid_points, valid_targets))

train_dataset = train_dataset.shuffle(len(train_points)).map(augment).batch(128)
valid_dataset = valid_dataset.shuffle(len(valid_points)).map(augment).batch(128)

print('Dataset splitted!')
print(f'train: {len(train_dataset)}\nvalidation: {len(valid_dataset)}')


points = 0
targets = 0
train_points = 0
valid_points = 0
train_targets = 0
valid_targets = 0
gc.collect()


### Defining model ###

def PotNet(num_filters):
    
    inputs = tf.keras.Input(shape=(1024, 3))
    
    # hidden layer
    conv1 = layers.Conv1D(64, kernel_size=1, padding="valid", activation='relu')(inputs)
    conv2 = layers.Conv1D(64, kernel_size=1, padding="valid", activation='relu')(conv1)
    conv3 = layers.Conv1D(64, kernel_size=1, padding="valid", activation='relu')(conv2)
    conv4 = layers.Conv1D(128, kernel_size=1, padding="valid", activation='relu')(conv3)
    conv5 = layers.Conv1D(1024, kernel_size=1, padding="valid", activation='relu')(conv4)

    pool = layers.GlobalMaxPooling1D()(conv5)


    # fully connected layer 1
    dense1 = layers.Dense(512)(pool)
    dense1 = layers.BatchNormalization(momentum=0.0)(dense1)
    dense1 = layers.Activation("relu")(dense1)

    drop = layers.Dropout(0.3)(dense1)

    # fully connected layer 2
    dense2 = layers.Dense(256)(drop)
    dense2 = layers.BatchNormalization(momentum=0.0)(dense2)
    dense2 = layers.Activation("relu")(dense2)

    drop = layers.Dropout(0.3)(dense2)

    #outputs (fully connected 3 / decision layer)
    outputs = layers.Dense(num_filters, activation=None)(drop)

    return tf.keras.Model(inputs=inputs, outputs=outputs, name="PotNet")


def euclidean_distance_loss(y_true, y_pred):
    """
    Euclidean distance loss
    https://en.wikipedia.org/wiki/Euclidean_distance
    :param y_true: TensorFlow/Theano tensor
    :param y_pred: TensorFlow/Theano tensor of the same shape as y_true
    :return: float
    """
    return K.sqrt(K.sum(K.square(y_pred - y_true), axis=-1))


### Compiling ###
if net == 1:
  model = PotNet(2)
else:
  model = PotNet(6)

model.compile(
  loss=euclidean_distance_loss,
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  metrics=[euclidean_distance_loss],
)

### Training the model ###

#early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=100)

checkpoint = tf.keras.callbacks.ModelCheckpoint(f'best_{net_dict[net]}_model.hdf5', monitor='val_loss', verbose=1,
    save_best_only=True, save_weights_only=False, mode='auto', save_freq='epoch')

history_logger = tf.keras.callbacks.CSVLogger(f'history_{net_dict[net]}.csv', separator=",", append=True)

if net == 1:
  history = model.fit(train_dataset, epochs=1000, validation_data=valid_dataset, callbacks=[checkpoint, history_logger], verbose=1)
else:
  history = model.fit(train_dataset, epochs=1500, validation_data=valid_dataset, callbacks=[checkpoint, history_logger], verbose=1)

print('Success!')



