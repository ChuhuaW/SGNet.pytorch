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

from lib.utils.eval_utils import eval_hevi


def train(model, train_gen, criterion, optimizer, device):
    model.train() # Sets the module in training mode.

    total_goal_loss = 0
    total_dec_loss = 0
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):
            input_traj, input_flow, target_traj = data
            batch_size = input_traj.shape[0]
            #print(batch_size)
            input_traj = input_traj.to('cuda', non_blocking=True)
            input_flow = input_flow.to('cuda', non_blocking=True)
            target_traj = target_traj.to('cuda', non_blocking=True)

            all_goal_traj, all_dec_traj = model([input_traj,input_flow])
            goal_loss = criterion(all_goal_traj, target_traj)
            dec_loss = criterion(all_dec_traj, target_traj)

            train_loss = goal_loss + dec_loss

            total_goal_loss += goal_loss.item()* batch_size
            total_dec_loss += dec_loss.item()* batch_size


            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
    total_goal_loss /= len(train_gen.dataset)
    total_dec_loss /= len(train_gen.dataset)

    
    return total_goal_loss, total_dec_loss, total_goal_loss + total_dec_loss


def test(model, test_gen, criterion, device):
    total_goal_loss = 0
    total_dec_loss = 0
    ADE_15 = 0
    ADE_05 = 0 
    ADE_10 = 0 
    FDE = 0 
    FIOU = 0
    CADE = 0 
    CFDE = 0
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):

            input_traj, input_flow, target_traj = data
            batch_size = input_traj.shape[0]
            input_traj = input_traj.to('cuda', non_blocking=True)
            input_flow = input_flow.to('cuda', non_blocking=True)
            target_traj = target_traj.to('cuda', non_blocking=True)

            all_goal_traj, all_dec_traj = model([input_traj,input_flow])


            goal_loss = criterion(all_goal_traj, target_traj)
            dec_loss = criterion(all_dec_traj, target_traj)

            test_loss = goal_loss + dec_loss

            total_goal_loss += goal_loss.item()* batch_size
            total_dec_loss += dec_loss.item()* batch_size

            all_dec_traj_np = all_dec_traj.to('cpu').numpy()
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()

            # Decoder
            batch_ADE_15, batch_ADE_05, batch_ADE_10, batch_FDE, batch_CADE, batch_CFDE, batch_FIOU =\
                eval_hevi(input_traj_np, target_traj_np, all_dec_traj_np)

            ADE_15 += batch_ADE_15
            ADE_05 += batch_ADE_05
            ADE_10 += batch_ADE_10
            FDE += batch_FDE
            CADE += batch_CADE
            CFDE += batch_CFDE
            FIOU += batch_FIOU
            

    
    ADE_15 /= len(test_gen.dataset)
    ADE_05 /= len(test_gen.dataset)
    ADE_10 /= len(test_gen.dataset)
    FDE /= len(test_gen.dataset)
    FIOU /= len(test_gen.dataset)
    
    CADE /= len(test_gen.dataset)
    CFDE /= len(test_gen.dataset)

    test_loss = total_goal_loss/len(test_gen.dataset) + total_dec_loss/len(test_gen.dataset)

    print("ADE_05: %4f;  ADE_10: %4f;  ADE_15: %4f;   FDE: %4f;   FIOU: %4f\n" % (ADE_05, ADE_10, ADE_15, FDE, FIOU))
    print("CFDE: %4f;   CADE: %4f;  \n" % (CFDE, CADE))
    return test_loss, ADE_15, ADE_05, ADE_10, FDE, FIOU, CADE, CFDE
