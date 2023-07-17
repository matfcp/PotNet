'''
Get MSE from predictions and actual outputs over synthetic datasets;
Get AE point-on-point for synthetic dataset test.
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
compile_nets(rot, trans)
#model.summary()

#model.load_model('best_model.hdf5')

print('\nAvailable vessels: VG, VM and VP')
vessel = input('Type the desired option: ')
rot.load_weights(f'/path/to/best_rot_model.hdf5')
trans.load_weights(f'/path/to/best_trans_model.hdf5')

path = f'/path/to/synthetic/test/set/'

ae = []

for folder in os.listdir(path):
  for test_file in os.listdir(path + folder):
    mesh = trimesh.load(path + folder + '/' + test_file)

    if type(mesh) == trimesh.base.Trimesh and mesh.bounding_box.volume > 0.00001:
      ms = pymeshlab.MeshSet()
      ms.load_new_mesh(path + folder + '/' + test_file)
      poisson_points = poisson(ms)

      r, theta, phi, yz = yz_normalization(poisson_points)
      canon = generate_canon(yz)

      pred_rot = rot.predict(canon[None, :, :])
      pred_trans = trans.predict(canon[None, :, :])
      #preds_rot.append(pred_rot[0])
      #preds_trans.append(pred_trans[0])
      T_pred = T_matrix(pred_rot[0], pred_trans[0])

      T = kabsch(yz, canon)
      actual_rot = np.concatenate((np.array(T[:3,:3][:,0]), np.array(T[:3,:3][:,1])))
      actual_trans = [T[1,3], T[2,3]]
      T_actual = T_matrix(actual_rot, actual_trans)

      # geting AE ptc
      points4dim = np.array([np.append(k, 1) for k in canon])
      pred_position = (points4dim @ T_pred.T)[:,:3]

      errors_xyz, std, mean = absolute_error_p2p(yz, pred_position, T_pred)
      ae.extend(errors_xyz)

# general mse and ae metrics
file = open(f'{vessel}_mse_ae.txt', 'w')
mse_T, rmse_T = mse_matrix(T_actual, T_pred)

file.write(f'\nMSE T: {mse_T:.7f}')
file.write(f'\nRMSE T: {rmse_T:.7f}')
file.write(f'\n\nae std deviation: {np.std(ae):.7f}')
file.write(f'\nae mean: {np.mean(ae, axis=0):.7f}')

p = np.percentile(ae, [25,50,75,100])
file.write(f'\nae percentiles (25, 50, 75, 100): {p[0]:.7f}, {p[1]:.7f}, {p[2]:.7f}, {p[3]:.7f}')

# how many of the errors are lower than some predefined values
four = [i for i in ae if abs(i) >= 0.04]
four = len(four) * 100 / len(ae)

two = [i for i in ae if abs(i) >= 0.02 and abs(i) < 0.04]
two = len(two) * 100 / len(ae)

one = [i for i in ae if abs(i) >= 0.01 and abs(i) < 0.02]
one = len(one) * 100 / len(ae)

half_one = [i for i in ae if abs(i) >= 0.005 and abs(i) < 0.01]
half_one = len(half_one) * 100 / len(ae)

half = [i for i in ae if abs(i) <= 0.005]
half = len(half) * 100 / len(ae)

print(four + two + one + half_one + half)

file.write('\n\nAbsolute errors')
file.write(f'\ne > 4 cm: {four}')
file.write(f'\n4 cm > e > 2 cm: {two}')
file.write(f'\n2 cm > e > 1 cm: {one}')
file.write(f'\n1 cm > e > 0.5 cm: {half_one}')
file.write(f'\ne < 0.5 cm: {half}')

# ae point on point plot
plt.title('Absolute errors point-on-point')
plt.ylabel('quantities')
plt.xlabel('measure in meters')
plt.xlim([-0.05, 0.05])
plt.hist(ae, bins=200, color='orange')
plt.savefig(f'{vessel}_ae_plot.png')



#
