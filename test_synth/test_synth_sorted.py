'''
Test sinthetic test datasets in alphabetical order.
'''

import os
import gc
import sys
import h5py
import math
import pickle
import random
import trimesh
import pymeshlab
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error as mse
# utils
sys.path.insert(1, '/path/to/utils')
from utils import *

# disable GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

rot = RegressioNet(6)
trans = RegressioNet(2)

rot.compile(
  loss=euclidean_distance_loss,
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  metrics=[euclidean_distance_loss],
)

trans.compile(
  loss=euclidean_distance_loss,
  optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
  metrics=[euclidean_distance_loss],
)

#model.summary()

#model.load_model('best_model.hdf5')

print('\nAvailable vessels: VG, VM and VP')
vessel = input('Type the desired option: ')
rot.load_weights(f'/path/to/best_rot_model.hdf5')
trans.load_weights(f'/path/to/best_trans_model.hdf5')

path = f'/path/to/synthetic/test/set/'

actual_rot = []
actual_trans = []
preds_rot = []
preds_trans = []
filename = []

directory = 'file2101'

for i in range(1, len(os.listdir(f'{path + directory}/')) + 1):
  mesh = trimesh.load(f'{path + directory}/{directory}_frac_{i}.stl')

  if type(mesh) == trimesh.base.Trimesh and mesh.bounding_box.volume > 0.00001:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(f'{path + directory}/{directory}_frac_{i}.stl')
    poisson_vertices = poisson(ms)

    r, theta, phi, zy = yz_normalization(poisson_points)
    canon = generate_canon(zy)
    pred_rot = rot.predict(canon[None, :, :])
    pred_trans = trans.predict(canon[None, :, :])
    preds_rot.append(pred_rot[0])
    preds_trans.append(pred_trans[0])

    T = kabsch(zy, canon)
    actual_rot.append(np.concatenate((np.array(T[:3,:3][:,0]), np.array(T[:3,:3][:,1]))))
    actual_trans.append([T[1,3], T[2,3]])
    filename.append(f'frac_{i}')

file = open(f'{vessel}_preds_{directory}.txt', 'w')
for i in range(len(actual_rot)):
  file.write(f'{filename[i]}\n')
  file.write(f'actual rot: {actual_rot[i]}\npred rot: {preds_rot[i]}\n')
  file.write(f'actual trans: {actual_trans[i]}\npred trans: {preds_trans[i]}\n')
  file.write(f'MSE rot: {mse(actual_rot[i], preds_rot[i])}\n')
  file.write(f'MSE trans: {mse(actual_trans[i], preds_trans[i])}\n\n')

# general error
file.write(f'\ngeneral MSE rot: {mse(actual_rot, preds_rot)}')
file.write(f'\ngeneral MSE trans: {mse(actual_trans, preds_trans)}')



#
