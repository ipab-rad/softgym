import argparse
import os
import os.path as osp
from progressbar import ProgressBar
import json
import matplotlib.pyplot as plt

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from progressbar import ProgressBar
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader

from torch.utils.tensorboard import SummaryWriter
    
parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default="data_intersect/")
parser.add_argument('--out_dir', type=str, default="output/")
parser.add_argument('--batch_size', type=int, default=4)
parser.add_argument('--latent_n', type=int, default=2)
parser.add_argument('--lr', type=float, default=1e-2)
parser.add_argument('--wd', type=float, default=1e-5)
parser.add_argument('--grad_clip', type=float, default=5e-2)
parser.add_argument('--epochs', type=int, default=3)
parser.add_argument('--use_gpu', type=int, default=1)
parser.add_argument('--save_runs', type=int, default=1)

class IntersectDataset(Dataset):
    def __init__(self, args, data_path="", norm=True):

        self.data = np.load(data_path + ".npy").astype(np.float32)
        self.labels = np.load(data_path + "_labels.npy").astype(np.float32)

    def __len__(self):

        return len(self.data)

    def __getitem__(self, idx):

        points = self.data[idx]
        label = self.labels[idx]

        return points, label

class IntersectMLP(nn.Module):
    
    def __init__(self, in_channels_n=18, latent_n=1, device="cpu"):
        super(IntersectMLP, self).__init__()
        layers = []
        self.latent_n = latent_n
        self.device = device

        self.dense_channels = [256, 256, self.latent_n]
        
        # ENCODER
        self.encoder_dense_0 = nn.Linear(in_channels_n, self.dense_channels[0])
        self.encoder_dense_1 = nn.Linear(self.dense_channels[0], self.dense_channels[1])
        self.encoder_dense_2 = nn.Linear(self.dense_channels[1], self.dense_channels[2])
        
        self.encoder = [self.encoder_dense_0,
                        self.encoder_dense_1,
                        self.encoder_dense_2]
        
        self.init_weights()
        
    def init_weights(self):
        for i in range(len(self.encoder)):
            self.encoder[i].weight.data.normal_(0, 0.01)
        
    def encode(self, x):

        x = torch.flatten(x, start_dim=1, end_dim=2)

        dense_0_encoded = F.leaky_relu(self.encoder_dense_0(x))
        dense_1_encoded = F.leaky_relu(self.encoder_dense_1(dense_0_encoded))
        y = self.encoder_dense_2(dense_1_encoded)
        # y = torch.sigmoid(self.encoder_dense_2(dense_1_encoded))

        return y


if __name__ == "__main__":
    
    args = parser.parse_args()
    
    train_dataset = IntersectDataset(args, data_path=osp.join(args.data_dir, 'train'))
    valid_dataset = IntersectDataset(args, data_path=osp.join(args.data_dir, 'valid'))

    print("Train dataset shape", np.array(train_dataset.data).shape)
    print("Valid dataset shape", np.array(valid_dataset.data).shape)


    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    valid_loader = DataLoader(dataset=valid_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    loaders = {'train': train_loader, 'valid' : valid_loader}
    
    device = None
    if args.use_gpu:
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    model = IntersectMLP(device=device)
    model.to(device)
    
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.wd)

    from torch.optim.lr_scheduler import ReduceLROnPlateau
    scheduler = ReduceLROnPlateau(optimizer, 'min', patience=5, threshold=1e-2, 
                                                    factor=0.25, min_lr=1e-3)
    
    phases = ['train', 'valid']
    stats = {}

    from datetime import datetime
    timestamp = str(datetime.now()).split('.')[0]

    if args.save_runs:
        # comment="_{0}_{1}seq_{2}y".format(PATH.split('/')[-1].lower(), args.seq_len, args.latent_n)
        # logdir = osp.join("runs/", args.out_dir.split('/')[-1], timestamp + comment)
        logdir = osp.join("runs/", args.out_dir.split('/')[-1], timestamp)
        writer = SummaryWriter(log_dir=logdir)
    
    for epoch in range(args.epochs):

        stats[epoch] = {}
        
        for phase in phases:
            
            loader = loaders[phase]
            
            stats[epoch][phase] = {'loss': 0, 'normalizer': len(loader)}
            
            model.train(phase == 'train')

            bar = ProgressBar(max_value=len(loader))

            for i, (x, y) in bar(enumerate(loader)):

                with torch.set_grad_enabled(phase == 'train'):
                    
                    if args.use_gpu:
                        x = x.cuda()
                        y = y.cuda()

                    bs = len(x)

                    y_hat = model.encode(x)

                    # loss = ((y - p) ** 2).mean()
                    loss = 100 * F.l1_loss(y[:, None], y_hat)

                    stats[epoch][phase]['loss'] += loss.item() / stats[epoch][phase]['normalizer']
                    
                if phase == 'train':
                    optimizer.zero_grad()
                    loss.backward()
                    nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                    optimizer.step()

            if phase == 'valid':
                scheduler.step(stats[epoch]['train']['loss'])
                print('Epoch:', epoch,'LR:', scheduler._last_lr, "# bad epochs:", scheduler.num_bad_epochs)
                if args.save_runs:
                    writer.add_scalar('Learning rate', scheduler._last_lr[0], epoch)

            
            if args.save_runs:
                writer.add_scalar('loss/'+phase, stats[epoch][phase]['loss'], epoch)

        print(("Epoch: {0}  " + \
               "T_loss: {1}  V_loss: {2}\n").format(epoch, 
                                                    round(stats[epoch]['train']['loss'], 5),
                                                    round(stats[epoch]['valid']['loss'], 5)))
        

    print('Finished Training.')

    PATH = osp.join(args.out_dir, timestamp)
    os.mkdir(PATH)

    torch.save(model.state_dict(), osp.join(PATH, "IntersectMLP_{0}.model".format(PATH.split('/')[1].lower())))
    
    print('Model Saved.')

    with open(osp.join(PATH, 'log.json'), 'w') as outputFile:
        json.dump(stats, outputFile)
        print('Log Saved.')

    with open(osp.join(PATH, 'args.json'), 'w') as outputFile:
        config = vars(args)
        json.dump(config, outputFile, indent=2)
        print('Args Saved.')

    layers = {}
    layers['dense_channels'] = model.dense_channels
    with open(osp.join(PATH, "layers.json"), 'w') as outputFile:
        json.dump(layers, outputFile, indent=2)
        print('Layers Saved.')


    # time python3 src/train_param.py --actuated 1 --rollout_len 128 --seq_len 64 --batch_size 32 --latent_n 3 --net_type mlp --save_runs 1 --epochs 100
    # python3 src/evaluate_actuated_UQ_param.py --data_dir ./data/param/active_frictionless --rollout_len 128 --seq_len 65 --latent_n 3 --net_type mlp --exp_idxs -1
    # python3 src/generate_pendulum_data_param.py --base_dir ./data/param --actuated 1 --rollout_len 128 --group_size 100 --actuation_freq 1 --param_n 3