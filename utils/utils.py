import os
import math
import random
import trimesh
import pymeshlab
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from sklearn.metrics import mean_squared_error as mse

def norm(v):
    return v/np.linalg.norm(v)

# Rotation matrix Y-axis:
def rotation_y(phi):
  return np.array([[np.cos(phi), 0, np.sin(phi)], [0, 1, 0], [-np.sin(phi), 0, np.cos(phi)]])


# Poisson distribution sampling
def poisson(ms):
  n = 1024
  #print(len(ms.current_mesh().vertex_matrix()))
  vertices = ms.current_mesh().vertex_matrix()
  if len(vertices) < 1024:
    while len(vertices) < 1024:
      n += 100
      ms.set_current_mesh(0)
      ms.generate_sampling_poisson_disk(samplenum=n)
      vertices = ms.current_mesh().vertex_matrix()

  while len(vertices) != 1024:
    try:
      index = random.randint(0, len(vertices))
      vertices = np.delete(vertices, index, axis=0)
    except:
      pass

  return vertices

'''
# Poisson distribution sampling
def poisson(ms):
  n = 2048
  
  ms.set_current_mesh(0)
  ms.generate_sampling_poisson_disk(samplenum=n)
  vertices = ms.current_mesh().vertex_matrix()
  
  shuflle = np.random.shuffle(vertices)
  
  return vertices[:1024]
'''

# Decimate sherds
def decimate(ms):

  target = ms.current_mesh().vertex_number() // 3000
  #target = 1024
  faces = 10 + 2 * target

  while(ms.current_mesh().vertex_number() > target):
    ms.meshing_decimation_quadric_edge_collapse(targetfacenum=faces)
    print('Decimated to: ', ms.current_mesh().face_number(), 'faces', ms.current_mesh().vertex_number(), 'vertices')
    faces -= (ms.current_mesh().vertex_number() - target)


# Generate yz normalization
def yz_normalization(poisson_vertices):
  '''
  Moves the poisson vertices of the mesh to the yz plane;
  Return: r, theta and points on yz normalized position.
  '''
  x, y, z = poisson_vertices.mean(axis=0)
  r = math.sqrt(x**2 + y**2 + z**2)
  theta = np.arccos(z/r)
  phi = np.arctan2(x, z)
  rot_yz = poisson_vertices @ rotation_y(phi)
  yz = rot_yz.astype(np.float32)

  # r, theta and phi
  #x, y, z = rot_yz.mean(axis=0)
  #r = math.sqrt(x**2 + y**2 + z**2)
  #theta = np.arccos(z/r)
  #phi = np.arctan2(x, z)

  return r, theta, phi, yz

# Generate canon (SVD):
def generate_canon(yz_points):
  '''
  Gets the canon position from points yz-normalized;
  Return: points in canon position at the origin.
  '''
  mesh_yz = trimesh.PointCloud(yz_points).convex_hull

  vc = mesh_yz.bounding_box_oriented.vertices - np.mean(mesh_yz.bounding_box_oriented.vertices, axis=0)
  U, S, Vt = np.linalg.svd(vc)

  centroid = np.mean(yz_points, axis=0)
  canon = yz_points - centroid
  canon = canon @ Vt.T

  return canon

# Kabsch algorithm (source -> target):
def kabsch(target, source):
  '''
  Takes 2 np.arrays and takes from source -> target.
  Returns: 4x4 transformation matrix.
  '''
  A = target
  B = source

  centroid_A = np.mean(A, axis=0)
  centroid_B = np.mean(B, axis=0)

  # center the points
  #AA = A - centroid_A
  #BB = B - centroid_B

  H = B.T @ A
  U, S, Vt = np.linalg.svd(H)

  R = Vt.T @ U.T
  t = -R @ centroid_B.T + centroid_A.T

  match_target = np.zeros((4,4))
  match_target[:3,:3] = R
  match_target[0,3] = t[0]
  match_target[1,3] = t[1]
  match_target[2,3] = t[2]
  match_target[3,3] = 1

  return match_target


# Network definition

def PotNet(n_filters):

    inputs = tf.keras.Input(shape=(1024, 3))

    # FEATURE EXTRACTION SUBNETWORK
    # hidden layer
    conv1 = layers.Conv1D(64, kernel_size=1, padding="valid", activation='relu')(inputs)
    conv2 = layers.Conv1D(64, kernel_size=1, padding="valid", activation='relu')(conv1)
    conv3 = layers.Conv1D(64, kernel_size=1, padding="valid", activation='relu')(conv2)
    conv4 = layers.Conv1D(128, kernel_size=1, padding="valid", activation='relu')(conv3)
    conv5 = layers.Conv1D(1024, kernel_size=1, padding="valid", activation='relu')(conv4)

    pool = layers.GlobalMaxPooling1D()(conv5)


    # OBJECT CLASSIFICATION SUBNETWORK
    # fully connected layer 1
    dense1 = layers.Dense(512)(pool)
    dense1 = layers.BatchNormalization(momentum=0.0)(dense1)
    dense1 = layers.Activation("relu")(dense1)

    # fully connected layer 2
    dense2 = layers.Dense(256)(dense1)
    dense2 = layers.BatchNormalization(momentum=0.0)(dense2)
    dense2 = layers.Activation("relu")(dense2)

    drop = layers.Dropout(0.3)(dense2)

    outputs = layers.Dense(n_filters, activation='linear')(drop)

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

def compile_nets(rot, trans):
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

    print('Models compiled!')


def T_matrix(pred_rot, pred_trans):
  '''
  Uses Gram-Schmidt orthogonalization to
  generate a T matrix from predictions.
  Returns: np.array 4x4 transformation matrix
  '''
  pred_rot = np.array(pred_rot)
  pred_trans = np.array(pred_trans)
  
  pred = pred_rot.reshape(2,3).T
  a1 = pred[:,0]
  a2 = pred[:,1]
  b1 = norm(a1)
  b2 = norm((a2-np.dot(b1,a2)*b1))
  b3 = np.cross(b1,b2)
  R = np.vstack([b1,b2,b3]).T

  t = pred_trans
  T = np.zeros((4,4))
  T[:3,:3] = R
  T[0,3] = 0
  T[1,3] = t[0]
  T[2,3] = t[1]
  T[3,3] = 1

  return T

def T_phi(phi, t):
  '''
  Gets a transformation matrix from phi that takes
  from pred position to real position.
  Returns: np.array 4x4 transformation matrix
  '''
  R = rotation_y(phi)
  T = np.zeros((4,4))
  T[:3,:3] = R
  T[0,3] = 0
  T[1,3] = t[0]
  T[2,3] = t[1]
  T[3,3] = 1

  return T

##################### QUANTITATIVE METRICS ##########################

def absolute_error_p2p(yz, pred_position, T_pred):
  '''
  Gives the absolute error point-to-point from yz normalized
  points set to the same points set transformed by the predicted matrix.
  Returns: array containing the absolute errors of x, y and z,
           standard deviation of errors and mean.
  '''

  errors_xyz = []

  for i in range(len(pred_position)):
        error = pred_position[i] - yz[i]
        for e in error:
          errors_xyz.append(e)

  return errors_xyz, np.std(errors_xyz), np.mean(errors_xyz, axis=0)

def mse_matrix(T_actual, T_pred):
  '''
  Gives the MSE from the actual and predicted transformation matrix.
  Returns: MSE matrix, RMSE matrix
  '''
  mse_T = mse(np.array(T_actual), np.array(T_pred))

  return mse_T, np.sqrt(mse_T)


def mse_ptc(yz, canon_transformed):
  '''
  Gets the the MSE from the point clouds in actual and predicted positions.
  Also, gets the RMSE penalization by coordinate (wich coordinate is most
  penalized by the models).
  Returns: MSE, RMSE and std. deviation from the two point clouds rounded to
  7 floating point number..
  '''
  errors = []
  error = (np.array(yz) - np.array(canon_transformed))**2
  mean_error = error.mean(axis=0)

  for e in error:
    errors.extend(e)

  mean = np.mean(errors, axis=0)
  rmse_mean_error = np.sqrt(mean_error)

  # putting in a tuple to visualize better in the table
  tuple_rmse = (round(rmse_mean_error[0], 7), round(rmse_mean_error[1], 7), round(rmse_mean_error[2], 7))

  return mean, np.sqrt(mean), np.std(errors), tuple_rmse


def Hausdorff_dist(yz, canon):
  '''
  Gives the Hausdorff distance between two points sets.
  Returns: min dist, max dist, std deviation of dist, mean of dist.
  '''

  distances = []
  for i in range(len(yz)):
      dist_min = 1000.0
      for j in range(len(canon)):
          dist = np.linalg.norm(yz[i] - canon[j])
          if dist_min > dist:
              dist_min = dist
      distances.append(dist_min)
  return max(distances), min(distances), np.std(distances), np.sqrt(np.mean(np.square(distances)))


def get_metrics(filename, dict_metrics, metrics=[], yz=None, pred_position=None, T_actual=None, T_pred=None):

  dict_metrics['filename'].append(filename)

  if 'ae' in metrics:
    errors_xyz, std_deviation, mean = absolute_error_p2p(yz, pred_position, T_pred)
    dict_metrics['stdAE'].append(std_deviation)
    dict_metrics['meanAE'].append(mean)

  if 'mse_matrix' in metrics:
    # matrix mse
    mse_T, rmse_T = mse_matrix(T_actual, T_pred)
    dict_metrics['mseT'].append(round(mse_T,7))
    dict_metrics['rmseT'].append(round(rmse_T,7))

  if 'mse_ptc' in metrics:
    # point cloud mse
    mserror, rmse_ptc, std_ptc, rmse_ptc_xyz = mse_ptc(yz, pred_position)
    dict_metrics['msePTC'].append(round(mserror,7))
    dict_metrics['rmsePTC'].append(round(rmse_ptc,7))
    dict_metrics['stdPTC'].append(round(std_ptc,7))
    dict_metrics['rmsePTC (x,y,z)'].append(rmse_ptc_xyz)

  if 'hausdorff' in metrics:
    # Hausdorff distance
    distHausd, minHausd, stdHausd, rmsHausd = Hausdorff_dist(yz, pred_position)
    dict_metrics['hausdDist'].append(distHausd)
    dict_metrics['hausdMin'].append(minHausd)
    dict_metrics['hausdStd'].append(stdHausd)
    dict_metrics['hausdRMS'].append(rmsHausd)



#

