'''
Get MSE ptc, MSE T and AE for the whole scanned dataset at once.
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

rot = PotNet(6)
trans = PotNet(2)
compile_nets(rot, trans)
#model.summary()

print('\nAvailable vessels: VM or VP\n')
vessel = input('Type the desired option: ')

rot.load_weights(f'/path/to/best_rot_model.hdf5')
trans.load_weights(f'/path/to/best_trans_model.hdf5')

path_target = f'/path/to/reassembled/decimated/target/shards/'
path_source = f'/path/to/reassembled/decimated/source/shards/'

mesh_tuple = 0
files = os.listdir(path_target)
errors = []
errors2 = []
errors_T = []
ae = []

for file in files:
	filename = file.split('.')[0]
	print(filename)

	# poisson -> canon
	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(path_source + file)
	poisson_source = poisson(ms)
	canon = generate_canon(poisson_source)

	pred_rot = rot.predict(canon[None, :, :], verbose=0)[0]
	pred_trans = trans.predict(canon[None, :, :], verbose=0)[0]
	T_pred = T_matrix(pred_rot, pred_trans)

	ms = pymeshlab.MeshSet()
	ms.load_new_mesh(path_target + file)
	poisson_target = poisson(ms)
	r, theta, phi, yz_target = yz_normalization(poisson_target)
	T = kabsch(yz_target, canon)

	points4dim = np.array([np.append(k, 1) for k in canon])
	pred_position = (points4dim @ T_pred.T)[:,:3]

	for k in range(len(pred_position)):
		# squared error
		error = (yz_target[k] - pred_position[k])**2
		errors.append(error[0])
		errors.append(error[1])
		errors.append(error[2])

		errors2.append(error)

		# absolute error
		abs_error = yz_target[k] - pred_position[k]
		ae.extend(abs_error)

	errors_T.append(mse(T, T_pred))

mse_pop = np.mean(errors, axis=0)
rmse_pop = np.sqrt(mse_pop)
std_pop = np.std(errors)

mse_vec = np.mean(errors2, axis=0)
rmse_vec = np.sqrt(mse_vec)

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


file = open(f'{vessel}_mse_general.txt', 'w')

file.write(f'\nMSE: {mse_pop:.7f}')
file.write(f'\nRMSE: {rmse_pop:.7f}')
file.write(f'\nRMSE (x,y,z): {rmse_vec[0]:.7f}, {rmse_vec[1]:.7f}, {rmse_vec[2]:.7f}')
file.write(f'\nstd: {std_pop:.7f}')

file.write(f'\nMSE T: {np.mean(errors_T, axis=0):.7f}')
file.write(f'\nRMSE T: {np.sqrt(np.mean(errors_T, axis=0)):.7f}')

file.write(f'\nMAE: {np.mean(ae, axis=0):.7f}')
file.write(f'\nstdAE: {np.std(ae):.7f}')
p = np.percentile(ae, [25,50,75])
file.write(f'\nAE percentiles (25, 50, 75): {p[0]:.7f}, {p[1]:.7f}, {p[2]:.7f}')

file.write('\n\nAbsolute errors')
file.write(f'\ne > 4 cm: {four}')
file.write(f'\n4 cm > e > 2 cm: {two}')
file.write(f'\n2 cm > e > 1 cm: {one}')
file.write(f'\n1 cm > e > 0.5 cm: {half_one}')
file.write(f'\ne < 0.5 cm: {half}')

# ae point on point plot
plt.title('Absolute errors point-on-point')
plt.ylabel('quantities')
plt.xlabel('measures in meters')
plt.xlim([-0.08, 0.08])
plt.hist(ae, bins=200, color='orange')
plt.savefig(f'{vessel}_ae_plot.png')
#plt.show()

