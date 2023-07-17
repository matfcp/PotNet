'''
Get test result over shards from a same break.
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

rot = PotNet(6)
trans = PotNet(2)

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

path = f'/path/to/test/set/'

rotate_y = int(input('Rotate over -phi?\n[1] Yes\n[2] No: '))

directory = 'file3101'
mesh_tuple = 0

for i in range(1, len(os.listdir(f'{path + directory}/')) + 1):
  mesh = trimesh.load(f'{path + directory}/{directory}_frac_{i}.stl')

  if type(mesh) == trimesh.base.Trimesh:
    ms = pymeshlab.MeshSet()
    ms.load_new_mesh(f'{path + directory}/{directory}_frac_{i}.stl')
    poisson_points = poisson(ms)

    r, theta, phi, yz = yz_normalization(poisson_points)
    canon = generate_canon(yz)
    T_unnorm2canon = kabsch(canon, poisson_points)

    pred_rot = rot.predict(canon[None, :, :])[0]
    pred_trans = trans.predict(canon[None, :, :])[0]
    T_pred = T_matrix(pred_rot, pred_trans)

    # get rotation y matrix over -phi
    T_rot_y = T_phi(-phi, [0,0])

    mesh.apply_transform(T_unnorm2canon)
    mesh.apply_transform(T_pred)
    if rotate_y == 1:
      mesh.apply_transform(T_rot_y)
    mesh_tuple += mesh

octants = trimesh.load('/path/to/octants.stl')
(mesh_tuple + octants).export(f'/path/to/save/{vessel}_synth_result.stl')

#

