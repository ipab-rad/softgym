import os
import os.path as osp
import numpy as np
import time as t # for .h5 data storing
import argparse

parser = argparse.ArgumentParser(description='Generate data.')
parser.add_argument('--unsplit_data_dir', type=str, default='data',
                    help='Path to unsplit dataset')
parser.add_argument('--split_data_dir', type=str, default='split_grid_cutting_data',
                    help='Path to split dataset')
parser.add_argument('--stat_file', type=str, default='stat.h5',
                    help='Statistics file of unsplit dataset')
parser.add_argument('--rollout_n', type=int, default=100,
                    help='How many rollouts/environments to split')
parser.add_argument('--train_valid_ratio', type=float, default=0.9,
                    help='What is the training - validation split ratio')
parser.add_argument('--train_dir', type=str, default='train',
                    help='Name of training folder')
parser.add_argument('--valid_dir', type=str, default='valid',
                    help='Name of validation folder')

args = parser.parse_args()
print(args)

# create train & valid directories
split_train_dir = os.path.join(args.split_data_dir, args.train_dir)
split_valid_dir = os.path.join(args.split_data_dir, args.valid_dir)
os.system('mkdir -p ' + split_train_dir)
os.system('mkdir -p ' + split_valid_dir)
# copy original statistics file
os.system('cp ' + os.path.join(args.unsplit_data_dir, args.stat_file) + ' ' + args.split_data_dir)

folders = [f for f in sorted(os.listdir(args.unsplit_data_dir)) if os.path.isdir(os.path.join(args.unsplit_data_dir, f))]
print(folders)

train_threshold = int(args.rollout_n * args.train_valid_ratio)
for f in folders:
    if f == "metrics":
        continue
    # copy training files as they are to the training folder
    if int(f) < train_threshold:
        os.system('cp -r ' + os.path.join(args.unsplit_data_dir, f) + ' ' + split_train_dir)
    # copy validation files in the validation folder, but first renaming them
    else:
        os.system('cp -r ' + os.path.join(args.unsplit_data_dir, f) + ' ' + os.path.join(split_valid_dir, str(int(f) - train_threshold)))
