
import os
import os.path as osp
import argparse
import numpy as np
from tqdm import tqdm
import h5py
import json
from scipy.spatial.transform import Rotation
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import cm


def check_intersect(line, plane):

        # https://en.wikipedia.org/wiki/Line%E2%80%93plane_intersection#Parametric_form

        la, lb = line
        p0, p1, p2 = plane

        lab = lb - la
        p01 = p1 - p0
        p02 = p2 - p0

        A = np.array([-lab, p01, p02])
        B = np.array(la - p0)

        try:
            A_inv = np.linalg.inv(A.T)
        except:
            return False, False

        t, u, v = np.matmul(A_inv, B.T).T

        return True, (t >= 0 and t <= 1 and u >= 0 and u <= 1 and v >= 0 and v <= 1 and (u+v) <= 1)


def sample_line_points(args, spawning_box):

    p1 = np.zeros(3)
    p1[0] = np.random.uniform(low=spawning_box['x'][0], high = spawning_box['x'][1])
    p1[1] = np.random.uniform(low=spawning_box['y'][0], high = spawning_box['y'][1])
    p1[2] = np.random.uniform(low=spawning_box['z'][0], high = spawning_box['z'][1])

    p2 = np.zeros(3)
    p2[0] = np.random.uniform(low=p1[0] - 2*args.p_radius, high = p1[0] + 2*args.p_radius)
    p2[1] = np.random.uniform(low=p1[1] - 2*args.p_radius, high = p1[1] + 2*args.p_radius)
    p2[2] = np.random.uniform(low=p1[2] - 2*args.p_radius, high = p1[2] - 2*args.p_radius)

    return p1, p2


def generate_data(args):

    train_data = []
    train_labels = []
    valid_data = []
    valid_labels = []

    # define the knife (currently a plane mesh object)
    knife_half_edge = np.array([0.0005, 0.2, 0.3])
    init_knife_offset_Y = (2 * knife_half_edge[1])
    spawning_box = {'x': [-0.5, 0.5], 'y': [0, 0.8], 'z': [-0.5, 0.5]}

    for knife_config_idx in range(args.n_knife_config):

        print("Knife configuration {0}/{1}".format(knife_config_idx, args.n_knife_config))

        data_chunk = []
        labels_chunk = []

        knife_center = np.zeros(3)
        knife_center[0] = np.random.uniform(low= -8*args.p_radius, high= 8*args.p_radius, size=1)
        knife_center[1] += np.random.rand() * init_knife_offset_Y
        knife_center[2] += np.random.uniform(low= -8*args.p_radius, high= 8*args.p_radius, size=1)

        knife_orientation = np.zeros(3)
        knife_orientation[1] = np.random.rand() * np.pi

        # quat = np.array([0., 0., 0., 1.])
        rot = Rotation.from_euler('xyz', knife_orientation)
        
        knife_plane = np.array([knife_center + np.array([0, knife_half_edge[1], knife_half_edge[2]]),
                                knife_center + np.array([0, knife_half_edge[1], -knife_half_edge[2]]),
                                knife_center + np.array([0, -knife_half_edge[1], knife_half_edge[2]]),
                                knife_center + np.array([0, -knife_half_edge[1], -knife_half_edge[2]])])

        # account for varying knife orientation when considering the knife plane
        # translate to origin - rotate - translate back to original position
        knife_plane -= knife_center
        knife_plane = rot.apply(knife_plane)
        knife_plane += knife_center

        knife_triangle_1 = np.array([knife_plane[0],
                                     knife_plane[2],
                                     knife_plane[3]])

        knife_triangle_2 = np.array([knife_plane[1],
                                     knife_plane[0],
                                     knife_plane[3]])

        # print("knife plane", knife_plane)

        positive_counter = 0
        negative_counter_around = 0
        negative_counter_below = 0

        pbar = tqdm(total = args.group_size)
        while positive_counter < args.group_size / 2:

            p1, p2 = sample_line_points(args, spawning_box)
            _, intersects_1 = check_intersect([p1, p2], knife_triangle_1)
            _, intersects_2 = check_intersect([p1, p2], knife_triangle_2)

            if intersects_1 or intersects_2:
                data_chunk.append([p1, p2, knife_plane[0], knife_plane[1], knife_plane[2], knife_plane[3]])
                labels_chunk.append(1)
                positive_counter += 1
                pbar.update(1)

        while negative_counter_around < args.group_size / 4:

            p1, p2 = sample_line_points(args, spawning_box)
            _, intersects_1 = check_intersect([p1, p2], knife_triangle_1)
            _, intersects_2 = check_intersect([p1, p2], knife_triangle_2)

            if not (intersects_1 and intersects_2):
                data_chunk.append([p1, p2, knife_plane[0], knife_plane[1], knife_plane[2], knife_plane[3]])
                labels_chunk.append(0)
                negative_counter_around += 1
                pbar.update(1)

        while negative_counter_below < args.group_size / 4:

            p1, p2 = sample_line_points(args, spawning_box)
            _, intersects_1 = check_intersect([p1, p2], knife_triangle_1)
            _, intersects_2 = check_intersect([p1, p2], knife_triangle_2)

            if intersects_1 or intersects_2:

                while intersects_1 or intersects_2:
                    s = np.random.randint(5) 
                    p1[1] -= s* args.p_radius
                    p2[1] -= s * args.p_radius
                    _, intersects_1 = check_intersect([p1, p2], knife_triangle_1)
                    _, intersects_2 = check_intersect([p1, p2], knife_triangle_2)

                data_chunk.append([p1, p2, knife_plane[0], knife_plane[1], knife_plane[2], knife_plane[3]])
                labels_chunk.append(0)
                negative_counter_below += 1
                pbar.update(1)

        train_n = int(args.data_split * args.group_size)
        train_indecies = np.random.choice(range(args.group_size), train_n, replace=False)
        valid_indecies = np.array(list(filter(lambda x : x not in train_indecies, range(args.group_size))))

        train_data += list(np.take(data_chunk, train_indecies, axis=0))
        train_labels += list(np.take(labels_chunk, train_indecies, axis=0))
        valid_data += list(np.take(data_chunk, valid_indecies, axis=0))
        valid_labels += list(np.take(labels_chunk, valid_indecies, axis=0))

        if args.visualise:

            ax = plt.axes(projection='3d')
            sur = np.array(data_chunk[0])
            # print(sur[2:5])

            ax.scatter(sur[2:6, 0], sur[2:6, 2], sur[2:6, 1], s=75, c='black', alpha=1, label='knife')
            ax.plot(sur[[2,3], 0], sur[[2,3], 2], sur[[2,3], 1], c='black', alpha=1)
            ax.plot(sur[[3,4], 0], sur[[3,4], 2], sur[[3,4], 1], c='black', alpha=1)
            ax.plot(sur[[4,5], 0], sur[[4,5], 2], sur[[4,5], 1], c='black', alpha=1)
            ax.plot(sur[[5,2], 0], sur[[5,2], 2], sur[[5,2], 1], c='black', alpha=1)
            
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            ax.set_zlabel("Y")

            label = 'cut'
            for i in range(0, args.group_size//2):
                p = np.array(data_chunk[i])
                ax.scatter(p[:2, 0], p[:2, 2], p[:2, 1], c='green', label=label)
                label = ''
                ax.plot(p[:2, 0], p[:2, 2], p[:2, 1], c='green')

            label = 'not cut 1'
            for i in range(args.group_size//2, args.group_size//4 * 3):
                p = np.array(data_chunk[i])
                ax.scatter(p[:2, 0], p[:2, 2], p[:2, 1], c='red', label=label)
                label = ''
                ax.plot(p[:2, 0], p[:2, 2], p[:2, 1], c='red')

            label = 'not cut 2'
            for i in range(args.group_size//4 * 3, args.group_size):
                p = np.array(data_chunk[i])
                ax.scatter(p[:2, 0], p[:2, 2], p[:2, 1], c='blue', label=label)
                label = ''
                ax.plot(p[:2, 0], p[:2, 2], p[:2, 1], c='blue')

            plt.legend()
            plt.show()

    train_data = np.array(train_data)
    train_labels = np.array(train_labels)
    valid_data = np.array(valid_data)
    valid_labels = np.array(valid_labels)

    print("\nTrain.shape", train_data.shape)
    print("Train positive", np.sum(train_labels == 1))
    print("Train negative", np.sum(train_labels == 0))
    print("Valid.shape", valid_data.shape)
    print("Valid positive", np.sum(valid_labels == 1))
    print("Valid negative", np.sum(valid_labels == 0))

    np.save(osp.join(args.data_dir, 'train.npy'), train_data)
    np.save(osp.join(args.data_dir, 'train_labels.npy'), train_labels)
    np.save(osp.join(args.data_dir, 'valid.npy'), valid_data)
    np.save(osp.join(args.data_dir, 'valid_labels.npy'), valid_labels)

    print("Saved data successfully at {0}".format(args.data_dir))

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, default='./data_intersect/', help='Path to the saved data')
    parser.add_argument('--n_knife_config', type=int, default=5, help='Number of knife configurations')
    parser.add_argument('--group_size', type=int, default=5, help='Number of rollouts (positive and negative) per knife config')
    parser.add_argument('--data_split', type=float, default=0.8)
    parser.add_argument('--p_radius', type=float, default=0.025)

    parser.add_argument('--visualise', type=int, default=0)

    args = parser.parse_args()

    generate_data(args)