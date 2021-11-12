import sys
import os
import os.path as osp
import numpy as np
import time
import random
from tqdm import tqdm
import torch
from torch import nn, optim
from torch.nn import functional as F
from torch.utils import data

from lib.utils.eval_utils import eval_jaad_pie, eval_jaad_pie_cvae
from lib.losses import cvae, cvae_multi

def train(model, train_gen, criterion, optimizer, device):
    model.train() # Sets the module in training mode.
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _  = model(inputs=input_traj, map_mask=None, targets=target_traj)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            goal_loss = criterion(all_goal_traj, target_traj)

            train_loss = goal_loss + cvae_loss + KLD_loss.mean()

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
    total_goal_loss /= len(train_gen.dataset)
    total_cvae_loss/=len(train_gen.dataset)
    total_KLD_loss/=len(train_gen.dataset)
    
    return total_goal_loss, total_cvae_loss, total_KLD_loss

def val(model, val_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    model.eval()
    loader = tqdm(val_gen, total=len(val_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(inputs=input_traj, map_mask=None, targets=None,training=False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            

            goal_loss = criterion(all_goal_traj, target_traj)


            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

    val_loss = total_goal_loss/len(val_gen.dataset)\
         + total_cvae_loss/len(val_gen.dataset) + total_KLD_loss/len(val_gen.dataset)
    return val_loss

def test(model, test_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    MSE_15 = 0
    MSE_05 = 0 
    MSE_10 = 0 
    FMSE = 0 
    FIOU = 0
    CMSE = 0 
    CFMSE = 0
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):
            batch_size = data['input_x'].shape[0]
            input_traj = data['input_x'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(inputs=input_traj, map_mask=None, targets=None, training=False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)


            goal_loss = criterion(all_goal_traj, target_traj)

            test_loss = goal_loss + cvae_loss

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            cvae_dec_traj = cvae_dec_traj.to('cpu').numpy()
            batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE, batch_CMSE, batch_CFMSE, batch_FIOU =\
                eval_jaad_pie_cvae(input_traj_np, target_traj_np[:,-1,:,:], cvae_dec_traj[:,-1,:,:,:])
            MSE_15 += batch_MSE_15
            MSE_05 += batch_MSE_05
            MSE_10 += batch_MSE_10
            FMSE += batch_FMSE
            CMSE += batch_CMSE
            CFMSE += batch_CFMSE
            FIOU += batch_FIOU
            

    
    MSE_15 /= len(test_gen.dataset)
    MSE_05 /= len(test_gen.dataset)
    MSE_10 /= len(test_gen.dataset)
    FMSE /= len(test_gen.dataset)
    FIOU /= len(test_gen.dataset)
    
    CMSE /= len(test_gen.dataset)
    CFMSE /= len(test_gen.dataset)
    

    test_loss = total_goal_loss/len(test_gen.dataset) \
         + total_cvae_loss/len(test_gen.dataset) + total_KLD_loss/len(test_gen.dataset)
    return test_loss, MSE_15, MSE_05, MSE_10, FMSE, FIOU, CMSE, CFMSE


def weights_init(m):
    if isinstance(m, nn.Linear):
        m.weight.data.normal_(0.0, 0.001)
    elif isinstance(m, nn.Conv1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.Conv3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose1d):
        nn.init.normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose2d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.ConvTranspose3d):
        nn.init.xavier_normal_(m.weight.data)
        if m.bias is not None:
            nn.init.normal_(m.bias.data)
    elif isinstance(m, nn.BatchNorm1d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm2d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.BatchNorm3d):
        nn.init.normal_(m.weight.data, mean=1, std=0.02)
        nn.init.constant_(m.bias.data, 0)
    elif isinstance(m, nn.LSTM):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.LSTMCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRU):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
    elif isinstance(m, nn.GRUCell):
        for param in m.parameters():
            if len(param.shape) >= 2:
                nn.init.orthogonal_(param.data)
            else:
                nn.init.normal_(param.data)
