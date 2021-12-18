from data import PrepareDataset, get_normalized_adj
from network import STGCN, qf_STGCN
from optim import Quantum_SGD
import torch.nn as nn
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm


def train():
    net = qf_STGCN(num_nodes=207, num_features=1, num_timesteps_input=20, num_timesteps_output=1)
    net.cuda()
    loss_mse = nn.MSELoss()
    loss_l1 = nn.L1Loss()

    path = '/home/vsfh/dataset/PS'
    speed_matrix = pd.read_pickle(path + '/LA_METR_Dataset/la_speed_after_final')
    A = np.load(path + '/LA_METR_Dataset/Metr_LA_2012_A.npy')
    A_norm = torch.FloatTensor(get_normalized_adj(A)).cuda()
    train_dataloader, valid_dataloader, test_dataloader, _ = PrepareDataset(speed_matrix, BATCH_SIZE=32,
                                                                                    seq_len=20, pred_len=5,
                                                                                    train_propotion=0.8,
                                                                                    valid_propotion=0.1)
    epochs = 500
    # optimizer = torch.optim.Adam(net.parameters(), lr=0.0001)
    optimizer = Quantum_SGD(net.parameters(), lr=0.0001)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=epochs)
    valid_loss = 100

    for i in range(epochs):
        for data in tqdm(train_dataloader):
            input, label = data
            #
            input = torch.FloatTensor(input).unsqueeze(0).permute(1,3,2,0).cuda()
            label = torch.FloatTensor(label).permute(0,2,1).cuda()

            optimizer.zero_grad()

            output = net(A_norm, input)
            loss = loss_mse(output, label)
            loss.backward()
            optimizer.step()
            scheduler.step()
        # validation
        with torch.no_grad():
            net.eval()
            this_val = 0
            for val in tqdm(valid_dataloader):
                input, label = val
                #
                input = torch.FloatTensor(input).unsqueeze(0).permute(1, 3, 2, 0).cuda()
                label = torch.FloatTensor(label).permute(0, 2, 1).cuda()

                optimizer.zero_grad()

                output = net(A_norm, input)
                loss = loss_mse(output, label)
                this_val += loss.cpu().numpy()
            print(this_val)
            if this_val < valid_loss:
                valid_loss = this_val
                torch.save(net.state_dict(), f'./training/offt_net_{i}.ckpt')


            pass


if __name__ == '__main__':
    train()
