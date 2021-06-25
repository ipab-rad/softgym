import argparse
import os
import os.path as osp
from progressbar import ProgressBar
import json
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits import mplot3d

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
from train_intersect_net import IntersectMLP


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--data_dir', type=str, default='./data_intersect/', help='Path to the saved data')
    parser.add_argument('--n_knife_config', type=int, default=5, help='Number of knife configurations')
    parser.add_argument('--group_size', type=int, default=5, help='Number of rollouts (positive and negative) per knife config')
    parser.add_argument('--data_split', type=float, default=0.8)
    parser.add_argument('--p_radius', type=float, default=0.025)

    parser.add_argument('--visualise', type=int, default=0)

    args = parser.parse_args()


    data = np.load("data_intersect/valid.npy").astype(np.float32)
    labels = np.load("data_intersect/valid_labels.npy").astype(np.float32)
    labels = labels[:, None]

    print("Data.shape", data.shape)
    print("Labels.shape", labels.shape)

    model = IntersectMLP()

    exp_idx = -1
    exp_list = os.listdir("output")
    exp_list.sort()
    exp_list = [x for x in exp_list if "tmp" not in x]

    print(exp_list)
    print(exp_list[exp_idx])

    PATH = osp.join("output", exp_list[exp_idx])
    model_filenames = [osp.join(PATH, x) for x in os.listdir(PATH) if '.model' in x]

    model.load_state_dict(torch.load(osp.join(model_filenames[0])))

    n = len(data)
    data_split = np.array([args.data_split])
    m = args.group_size

    print("n, m", n, m)

    for i in range(0, n, m):

        data_chunk = data[i : i + m]
        labels_chunk = labels[i : i + m]

        y = model.encode(torch.tensor(data_chunk))
        y = y.detach().cpu().numpy()
        diff = y - labels_chunk

        fig, axs = plt.subplots(1,1)

        axs.plot(range(m), y, label="pred")
        axs.plot(range(m), labels_chunk, label="true")
        axs.grid()

        plt.legend()
        plt.show()



        ax = plt.axes(projection='3d')
        sur = np.array(data_chunk[20])

        ax.scatter(sur[2:6, 0], sur[2:6, 2], sur[2:6, 1], s=75, c='black', alpha=1, label='knife')
        ax.plot(sur[[2,3], 0], sur[[2,3], 2], sur[[2,3], 1], c='black', alpha=1)
        ax.plot(sur[[3,4], 0], sur[[3,4], 2], sur[[3,4], 1], c='black', alpha=1)
        ax.plot(sur[[4,5], 0], sur[[4,5], 2], sur[[4,5], 1], c='black', alpha=1)
        ax.plot(sur[[5,2], 0], sur[[5,2], 2], sur[[5,2], 1], c='black', alpha=1)
            
        ax.set_xlabel("X")
        ax.set_ylabel("Z")
        ax.set_zlabel("Y")

        for i in range(0, m):
            p = np.array(data_chunk[i])

            color = 'green'

            if abs(diff[i]) > 0.1:
                color = 'red'

            # if labels[i] == 0:
            #     color = 'red'

            ax.scatter(p[:2, 0], p[:2, 2], p[:2, 1], c=color)
            ax.plot(p[:2, 0], p[:2, 2], p[:2, 1], c=color)

        plt.legend()
        plt.show()


