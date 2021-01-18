import os
import os.path as osp
import numpy as np
import pyflex
import time
import argparse
import cv2
import json

parser = argparse.ArgumentParser(description='Play back generated data.')
parser.add_argument('--dir', type=str, default='data',
                    help='Folder containing generated data')
parser.add_argument('--store_frames', type=bool, default=0,
                    help='Store the frames for each rollout in the corresponding folder')

args = parser.parse_args()
print(args)

window_size = 500

pyflex.init(0, 1, window_size, window_size)

for rollout_folder in sorted(os.listdir(args.dir)):

    positions = np.load(osp.join(args.dir, rollout_folder, 'positions.npy'))
    velocities = np.load(osp.join(args.dir, rollout_folder, 'velocities.npy'))
    fingers_positions = np.load(osp.join(args.dir, rollout_folder, 'gripper_fingers_positions.npy'))
    with open(osp.join(args.dir, rollout_folder, 'params.json'), "r") as read_file:
        params = json.load(read_file)

    print("\nPlaying data from folder: {0} with parameters: {1}\n".format(osp.join(args.dir, rollout_folder), params))

    scene_params = [params['x_part_n'], params['y_part_n'], params['z_part_n'], 
                    params['clusterStiffness'], params['clusterPlasticThreshold'], 
                    params['clusterPlasticCreep']]

    pyflex.set_scene(6, scene_params, 0)

    # default gripper finger poses; resampled for each grasp
    halfEdge = np.array([0.15, 0.8, 0.15])
    center = np.array([0., 0., 0.])
    quat = np.array([0, 0., 0., 1.])

    pyflex.add_box(halfEdge, center, quat)
    pyflex.add_box(halfEdge, center, quat)

    camera_param = {'pos': np.array([0, 4, 3]),
                    'angle': np.array([0 * np.pi, -0.25 * np.pi, 0]),
                    'width': window_size,
                    'height': window_size}
    pyflex.set_camera_params(np.array([*camera_param['pos'], *camera_param['angle'], 
                                        camera_param['width'], camera_param['height']]))

    for r in range(positions.shape[0]):
        for i in range(positions.shape[1]):
            pyflex.set_positions(positions[r, i])
            pyflex.set_shape_states(fingers_positions[r, i])

            img = pyflex.render()

            if args.store_frames:
                width, height = camera_param['width'], camera_param['height']
                img = img.reshape(height, width, 4)[::-1, :, :3]  # Need to reverse the height dimension
                
                path = os.path.join(args.dir, rollout_folder, 'render_%d.png' % (r * positions.shape[1] + i))
                cv2.imwrite(path, img[:, :, [0, 1, 2]])

    #pyflex.clean()