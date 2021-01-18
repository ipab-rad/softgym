# modified version of https://github.com/YunzhuLi/PyFleX/blob/master/bindings/examples/test_RiceGrip.py

import os
import os.path as osp
import numpy as np
import pyflex
import time
import argparse
import cv2
import json

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Generate data.')
parser.add_argument('--output_dir', type=str, default='data',
                    help='Output folder for accumulating generated data')
parser.add_argument('--grip_n', type=int, default=1,
                    help='How many times to grasp the softbody')
parser.add_argument('--grip_len', type=int, default=100,
                    help='Length of each grip as timesteps')
parser.add_argument('--rollout_n', type=int, default=3,
                    help='How many rollouts/environments to generate')

args = parser.parse_args()
print(args)

dt = 1. / 120.
window_size = 500

dim_position = 4 # each particle's state is in the format [x,y,z,1/m]; m is the particle mass
dim_velocity = 3 # velocity for each particle [x,y,z]
dim_shape_state = 14 # for each gripper finger keep curr and prev states => 2 * [x,y,z,qx,qy,qz,qw]


def sample_gripper_config(grip_idx):
    
    angle = np.random.rand() * np.pi * 2.
    
    # gripper center offset
    x = 0
    z = 0

    open_gripper_dist = 2.5
    closed_gripper_dist = 0.7

    # how quickly to close/open gripper
    graps_rate = 5

    return x, z, angle, open_gripper_dist, closed_gripper_dist, graps_rate

def quatFromAxisAngle(axis, angle):
    # https://www.euclideanspace.com/maths/geometry/rotations/conversions/angleToQuaternion/examples/index.htm
    axis /= np.linalg.norm(axis)

    half = angle * 0.5
    w = np.cos(half)

    sin_theta_over_two = np.sin(half)
    axis *= sin_theta_over_two

    quat = np.array([axis[0], axis[1], axis[2], w])

    return quat

def calc_shape_states(t, gripper_config):
    x, z, a, open_gripper_dist, closed_gripper_dist, graps_rate = gripper_config

    finger_travel_dist = (open_gripper_dist - closed_gripper_dist) / 2.
    half_open_gripper_dist = open_gripper_dist / 2.

    states = np.zeros((2, dim_shape_state))

    e_0 = np.array([x + half_open_gripper_dist*np.cos(a), z - half_open_gripper_dist*np.sin(a)])
    e_1 = np.array([x - half_open_gripper_dist*np.cos(a), z + half_open_gripper_dist*np.sin(a)])

    # for the x axis right is positive, left is negative
    # for the z axis down is positive, up is negative
    offset = np.array([np.cos(a), -np.sin(a)]) * finger_travel_dist
    
    time = max(0., t) * graps_rate
    lastTime = max(0., t - dt) * graps_rate

    e_0_curr = e_0 - offset * np.sin(time)
    e_1_curr = e_1 + offset * np.sin(time)

    e_0_last = e_0 - offset * np.sin(lastTime)
    e_1_last = e_1 + offset * np.sin(lastTime)

    quat = quatFromAxisAngle(np.array([0., 1., 0.]), a)

    states[0, :3] = np.array([e_0_curr[0], 0.6, e_0_curr[1]])
    states[0, 3:6] = np.array([e_0_last[0], 0.6, e_0_last[1]])
    states[0, 6:10] = quat
    states[0, 10:14] = quat

    states[1, :3] = np.array([e_1_curr[0], 0.6, e_1_curr[1]])
    states[1, 3:6] = np.array([e_1_last[0], 0.6, e_1_last[1]])
    states[1, 6:10] = quat
    states[1, 10:14] = quat

    return states

def visualize_point_cloud(positions, idx):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(positions[idx, 0], positions[idx, 1], positions[idx, 2], alpha = 0.5)
    plt.show()

def calc_surface_idx(positions):
    point_tree = spatial.cKDTree(positions)
    neighbors = point_tree.query(positions, 40, n_jobs=-1)[1]
    surface_mask = np.zeros(positions.shape[0])

    pca = PCA(n_components=3)

    for i in range(len(neighbors)):
        pca.fit(positions[neighbors[i]])
        # print(i, pca.explained_variance_ratio_)
        if pca.explained_variance_ratio_[0] > 0.45:
            surface_mask[i] = 1

    surface_idx = np.nonzero(surface_mask)[0]

    print('surface idx', surface_idx.shape)

    return surface_idx

def rand_float(lo, hi):
    return np.random.rand() * (hi - lo) + lo


pyflex.init(0, 1, window_size, window_size)


for rollout_idx in range(args.rollout_n):

    os.system('mkdir -p ' + osp.join(args.output_dir, str(rollout_idx)))

    ### set scene
    # x, y, z: how many particles per dimension
    # clusterStiffness: 
    # clusterPlasticThreshold: [0.000005, 0.0001]
    # clusterPlasticCreep: [0.1, 0.3]
    x_part_n = y_part_n = z_part_n = 8

    # clusterStiffness: [0.3, 0.8]; 0.3 - soft; 0.7 hard
    clusterStiffness = 0.7

    # clusterPlasticThreshold: [0.00001, 0.01]; 0.01 - elastic; 0.00001 - plastic
    clusterPlasticThreshold = 0.0001

    # clusterPlasticCreep: [0.1, 0.3]
    clusterPlasticCreep = 0.2

    scene_params = np.array([x_part_n, y_part_n, z_part_n, 
                             clusterStiffness, clusterPlasticThreshold, clusterPlasticCreep])

    # index of the desired scene in the g_scenes vector in the pyflex_init() method from pyflex.cpp
    pyflex.set_scene(6, scene_params, 0)

    # default gripper finger poses; resampled for each grasp
    halfEdge = np.array([0.15, 0.8, 0.15])
    center = np.array([0., 0., 0.])
    quat = np.array([0, 0., 0., 1.])

    pyflex.add_box(halfEdge, center, quat)
    pyflex.add_box(halfEdge, center, quat)

    n_particles = pyflex.get_n_particles()
    n_shapes = pyflex.get_n_shapes()

    print("\nGenerating rollout\t\t#{0}".format(rollout_idx))
    print('clusterStiffness:\t\t{0}'.format(clusterStiffness))
    print('clusterPlasticThreshold:\t{0}'.format(clusterPlasticThreshold))
    print('clusterPlasticCreep:\t\t{0}'.format(clusterPlasticCreep))
    print('n_particles: {0}, n_shapes: {1}\n'.format(n_particles, n_shapes))

    params = {}
    params['x_part_n'] = x_part_n
    params['y_part_n'] = y_part_n
    params['z_part_n'] = z_part_n
    params['n_particles'] = n_particles
    params['clusterStiffness'] = clusterStiffness
    params['clusterPlasticThreshold'] = clusterPlasticThreshold
    params['clusterPlasticCreep'] = clusterPlasticCreep
    params['gripper'] = []

    positions = np.zeros((args.grip_n, args.grip_len, n_particles, dim_position))
    velocities = np.zeros((args.grip_n, args.grip_len, n_particles, dim_velocity))
    shape_states = np.zeros((args.grip_n, args.grip_len, n_shapes, dim_shape_state))

    for r in range(args.grip_n):
        
        gripper_config = sample_gripper_config(r)

        params['gripper'].append({})
        params['gripper'][-1]['gripper_x'] = gripper_config[0]
        params['gripper'][-1]['gripper_z'] = gripper_config[1]
        params['gripper'][-1]['gripper_angle'] = gripper_config[2]
        params['gripper'][-1]['gripper_open_dist'] = gripper_config[3]
        params['gripper'][-1]['gripper_closed_dist'] = gripper_config[4]
        params['gripper'][-1]['gripper_grasp_rate'] = gripper_config[5]

        for i in range(args.grip_len):
            shape_states_ = calc_shape_states(i * dt, gripper_config)
            pyflex.set_shape_states(shape_states_)

            positions[r, i] = pyflex.get_positions().reshape(-1, dim_position)

            # if i == 0:
            #     surface_idx = calc_surface_idx(positions[r, i, :, :3])
            #     visualize_point_cloud(positions[r, i, :, :3], range(1000))

            velocities[r, i] = pyflex.get_velocities().reshape(-1, dim_velocity)
            shape_states[r, i] = pyflex.get_shape_states().reshape(-1, dim_shape_state)

            pyflex.step()

    np.save(osp.join(args.output_dir, str(rollout_idx), 'positions.npy'), positions)
    np.save(osp.join(args.output_dir, str(rollout_idx), 'velocities.npy'), velocities)
    np.save(osp.join(args.output_dir, str(rollout_idx), 'gripper_fingers_positions.npy'), shape_states)
    with open(osp.join(args.output_dir, str(rollout_idx), 'params.json'), "w") as write_file:
        json.dump(params, write_file, indent=4)


    # camera_param = {'pos': np.array([0, 4, 3]),
    #                 'angle': np.array([0 * np.pi, -0.25 * np.pi, 0]),
    #                 'width': window_size,
    #                 'height': window_size}
    # pyflex.set_camera_params(np.array([*camera_param['pos'], *camera_param['angle'], camera_param['width'], camera_param['height']]))

    # for r in range(args.grip_n):
    #     for i in range(args.grip_len):
    #         pyflex.set_positions(positions[r, i])
    #         pyflex.set_shape_states(shape_states[r, i])

    #         img = pyflex.render()
