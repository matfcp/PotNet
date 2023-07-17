'''
Counts the number of shards bigger than the minimum volume (0.00001)
'''

import os
import trimesh

path = f'/path/to/train/dataset/'

lenght_path = len(os.listdir(path))
print('Path lenght: ', lenght_path)

count = 0

print('Starting...')
for i in os.listdir(path):
  for j in os.listdir(path + i):
    mesh = trimesh.load(f'{path + i}/{j}')

    if type(mesh) == trimesh.base.Trimesh and mesh.bounding_box.volume > 0.00001:
      count += 1

print('total number of validated shards: ', count)
