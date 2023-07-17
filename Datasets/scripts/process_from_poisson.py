'''
1. produces a poisson-disk sample for each shard from the train set;
2. produces the yz-normalized position and canon position for each
shard from the train set;
3. saves the matrices containing the points sets;
'''

import os
import sys
import pickle
import trimesh
import numpy as np
import pymeshlab
import random
import gc
# utils
sys.path.insert(1, '/path/to/utils')
from utils import *

print('\nAvailable vessels: VG, VM and VP\n')
vessel = input('Type the desired option: ')

min_volume = 0.00001
if vessel in ['VP']:
  min_volume = 0.000001

path = f'/path/to/{vessel}_dataset'
path = os.path.join(path, 'files/train')
lenght_path = len(os.listdir(path))
print('path lenght: ', lenght_path)
path = [os.path.join(path, f'file{i}') for i in range(1, lenght_path + 1)]

# Parsing:
canon_points = []
poisson_points = []
yz_points = []
targets = []
T = []

print('Processing...')
for i in range(1, lenght_path + 1):
  for j in range(1, len(os.listdir(path[i-1])) + 1):
    if i % 100 == 0 and j == 1:
      print(f'Reached file{i}')

    mesh = trimesh.load(f'{path[i-1]}/file{i}_frac_{j}.stl')

    if type(mesh) == trimesh.base.Trimesh and mesh.bounding_box.volume > min_volume:
      #print(f'file{i}_frac_{j}')
      ms = pymeshlab.MeshSet()
      ms.load_new_mesh(f'{path[i-1]}/file{i}_frac_{j}.stl')
      poisson_vertices = poisson(ms)
      poisson_points.append(poisson_vertices)

      r, theta, phi, yz = yz_normalization(poisson_vertices)
      canon = generate_canon(yz)

      yz_points.append(yz)
      canon_points.append(canon)
      targets.append([r, theta, phi])

      k = kabsch(yz, canon)
      T.append(k)


path_save = f'/path/to/save/train_files/'

array_file = open(path_save + 'poisson_points.npy', 'wb')
np.save(array_file, poisson_points)
array_file.close()

array_file = open(path_save + 'canon_points.npy', 'wb')
np.save(array_file, canon_points)
array_file.close()

array_file = open(path_save + 'yz_points.npy', 'wb')
np.save(array_file, yz_points)
array_file.close()

array_file = open(path_save + 'canon_targets.npy', 'wb')
np.save(array_file, targets)
array_file.close()

array_file = open(path_save + 'T.npy', 'wb')
np.save(array_file, T)
array_file.close()

