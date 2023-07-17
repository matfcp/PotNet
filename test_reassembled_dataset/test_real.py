'''
Test PotNet models over decimated (factor 3000) and scaled, yz-normalized shards
from rebuilt dataset;

Get evaluation metrics from the same point clouds in two different
positions: yz normalized and predicted position;
'''

import os
import sys
import h5py
import math
import json
import trimesh
import pymeshlab
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import layers

# utils
sys.path.insert(1, '/path/to/utils')

from utils import *

# disable GPU
#os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

rot = RegressioNet(6)
trans = RegressioNet(2)
compile_nets(rot, trans)
#model.summary()

print('\nAvailable vessels: VM or VP\n')
vessel = input('Type the desired option: ')

rotate_phi = int(input('\n\nRotate over -phi?\n[1] Sim\n[2] Não\n'))

rot.load_weights(f'/path/to/best_rot_model.hdf5')
trans.load_weights(f'/path/to/best_trans_model.hdf5')

path_target = f'/path/to/reassembled/decimated/target/shards/'
path_source = f'/path/to/reassembled/decimated/source/shards/'

mesh_tuple = 0
files = os.listdir(path_target)

metrics = {'filename': [], 'msePTC': [], 'rmsePTC': [], 'rmsePTC (x,y,z)': [], 'stdPTC': [], 'mseT': [], 'rmseT': []}
T_dict = {}

for file in files:
	filename = file.split('.')[0]
	print(filename)

	# poisson -> canon ->
	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(path_source + file)
	poisson_source = poisson(ms)

	# K (unnorm poisson -> canon)
	canon = generate_canon(poisson_source)
	K = kabsch(canon, poisson_source)

	# predict -> T pred
	pred_rot = rot.predict(canon[None, :, :], verbose=0)[0]
	pred_trans = trans.predict(canon[None, :, :], verbose=0)[0]
	T_pred = T_matrix(pred_rot, pred_trans)

	# unnormalized -> yz; canon -> pred position
	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(path_target + file)
	poisson_target = poisson(ms)
	r, theta, phi, yz_target = yz_normalization(poisson_target)

	# canon -> yz
	T_canon2yz = kabsch(yz_target, canon)

	# real -> yz
	T_real2yz = kabsch(yz_target, poisson_target)

	# pred -> real
	T_rot_phi = T_phi(-phi, [0,0])

	points4dim = np.array([np.append(k, 1) for k in canon])
	pred_position = (points4dim @ T_pred.T)[:,:3]

	# save predicted matrix
	T_dict[filename] = {'unnorm2canon': K.tolist(), 'T_pred': T_pred.tolist(),
				'real2yz': T_real2yz.tolist()}

	# get metrics
	get_metrics(filename, metrics, ['mse_ptc', 'mse_matrix'], yz_target, pred_position, T_canon2yz, T_pred)

	# visualizing result
	mesh_target = trimesh.load(path_target + file)
	mesh_target.apply_transform(T_real2yz)

	mesh_source = trimesh.load(path_source + file)
	mesh_source.apply_transform(K)
	mesh_source.apply_transform(T_pred)
	if rotate_phi == 1:
		mesh_source.apply_transform(T_rot_phi)
	mesh_tuple += mesh_source

octants = trimesh.load('/path/to/octants.stl')
#(mesh_tuple + octants).show()

save = input('Export result?\n[1] Yes\n[2] No\nType the desired option:  ')
if save == '1':
  (mesh_tuple + octants).export(f'/path/to/export')
  print('>>> Result exported!')
#mesh_tuple.show()

# exporting data to xlsx file
df = pd.DataFrame.from_dict(metrics)
df.sort_values('filename', inplace=True, ignore_index=True)
df.to_csv(f'/path/to/save/metrics/{vessel}_real_metrics.csv')

# exporting T matrices dict
with open(f'path/to/save/metrics/{vessel}_T_dict.txt','w') as js:
  json.dump(T_dict, js)
