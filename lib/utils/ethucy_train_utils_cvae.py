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

from lib.utils.eval_utils import eval_ethucy, eval_ethucy_cvae
from lib.losses import cvae, cvae_multi

def train(model, train_gen, criterion, optimizer, device):
    model.train() # Sets the module in training mode.
    count = 0
    total_goal_loss = 0
    total_dec_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    loader = tqdm(train_gen, total=len(train_gen))
    with torch.set_grad_enabled(True):
        for batch_idx, data in enumerate(loader):
            # if batch_idx > 1:
            #     break
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_traj_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(input_traj, map_mask = None, targets = target_traj, start_index = first_history_index, training =  False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj, first_history_index[0])
            #import pdb; pdb.set_trace()
            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])
            train_loss = goal_loss + cvae_loss  + KLD_loss.mean()

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

            # optimize
            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
        
    total_goal_loss /= count
    total_cvae_loss/= count
    total_KLD_loss/= count
    
    return total_goal_loss, total_cvae_loss, total_KLD_loss

def val(model, val_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    count = 0
    model.eval()
    loader = tqdm(val_gen, total=len(val_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):
            # if batch_idx > 1:
            #     break
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_traj_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(input_traj, map_mask = None, targets = None, start_index = first_history_index, training =  False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            

            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

    val_loss = total_goal_loss/count \
         + total_cvae_loss/count+ total_KLD_loss/ count
    #import pdb;pdb.set_trace()
    return val_loss

def test(model, test_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    ADE_08 = 0
    ADE_12 = 0 
    FDE_08 = 0 
    FDE_12 = 0 
    count = 0
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):
            # if batch_idx > 1:
            #     break
            
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_traj_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)

            all_goal_traj, cvae_dec_traj, KLD_loss, _ = model(input_traj, map_mask = None, targets = None, start_index = first_history_index, training =  False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])

            

            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

            cvae_dec_traj = cvae_dec_traj.to('cpu').numpy()
            all_goal_traj_np = all_goal_traj.to('cpu').numpy()
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()

            batch_results =\
                eval_ethucy_cvae(input_traj_np, target_traj_np[:,-1,:,:], cvae_dec_traj[:,-1,:,:,:])
            ADE_08 += batch_results['ADE_08']
            ADE_12 += batch_results['ADE_12']
            FDE_08 += batch_results['FDE_08']
            FDE_12 += batch_results['FDE_12']
            

    
    ADE_08 /= count
    ADE_12 /= count
    FDE_08 /= count
    FDE_12 /= count
    

    test_loss = total_goal_loss/count + total_cvae_loss/count + total_KLD_loss/count
    # print("Test Loss %4f\n" % (test_loss))
    # print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))
    return test_loss, ADE_08, FDE_08, ADE_12, FDE_12

def evaluate(model, test_gen, criterion, device):
    total_goal_loss = 0
    total_cvae_loss = 0
    total_KLD_loss = 0
    ADE_08 = 0
    ADE_12 = 0 
    FDE_08 = 0 
    FDE_12 = 0 
    count = 0
    all_file_name = []
    model.eval()
    loader = tqdm(test_gen, total=len(test_gen))
    with torch.set_grad_enabled(False):
        for batch_idx, data in enumerate(loader):#for batch_idx, data in enumerate(val_gen):            
            first_history_index = data['first_history_index']
            assert torch.unique(first_history_index).shape[0] == 1
            batch_size = data['input_x'].shape[0]
            count += batch_size
            
            input_traj = data['input_x'].to(device)
            input_traj_st = data['input_x_st'].to(device)
            target_traj = data['target_y'].to(device)
            scene_name = data['scene_name'] 
            timestep = data['timestep']
            current_img = timestep
            #import pdb; pdb.set_trace()
            # filename = datapath + '/test/biwi_eth.txt'
            # data = pd.read_csv(filename, sep='\t', index_col=False, header=None)
            # data.columns = ['frame_id', 'track_id', 'pos_x', 'pos_y']
            # frame_id_min = data['frame_id'].min()
            # filename path = os.path.join(datapath, dataset ,str((current_img[1][0]+int(frame_id_min)//10)*10).zfill(5) + '.png')

            all_goal_traj, cvae_dec_traj, KLD_loss = model(input_traj, target_traj, first_history_index, False)
            cvae_loss = cvae_multi(cvae_dec_traj,target_traj)
            goal_loss = criterion(all_goal_traj[:,first_history_index[0]:,:,:], target_traj[:,first_history_index[0]:,:,:])
            total_goal_loss += goal_loss.item()* batch_size
            total_cvae_loss += cvae_loss.item()* batch_size
            total_KLD_loss += KLD_loss.mean()* batch_size

            cvae_dec_traj_np = cvae_dec_traj.to('cpu').numpy()
            cvae_dec_traj = cvae_dec_traj.to('cpu').numpy()

            all_goal_traj_np = all_goal_traj.to('cpu').numpy()
            input_traj_np = input_traj.to('cpu').numpy()
            target_traj_np = target_traj.to('cpu').numpy()
            #import pdb;pdb.set_trace()
            # Decoder
            # batch_MSE_15, batch_MSE_05, batch_MSE_10, batch_FMSE, batch_CMSE, batch_CFMSE, batch_FIOU =\
            #     eval_jaad_pie(input_traj_np, target_traj_np, all_dec_traj_np)
            batch_results =\
                eval_ethucy_cvae(input_traj_np, target_traj_np[:,-1,:,:], cvae_dec_traj[:,-1,:,:,:])
            ADE_08 += batch_results['ADE_08']
            ADE_12 += batch_results['ADE_12']
            FDE_08 += batch_results['FDE_08']
            FDE_12 += batch_results['FDE_12']

            if batch_idx == 0:
                all_input = input_traj_np
                all_target = target_traj_np
                all_prediction = cvae_dec_traj_np
            else:
                all_input = np.vstack((all_input,input_traj_np))
                all_target = np.vstack((all_target,target_traj_np))
                all_prediction = np.vstack((all_prediction,cvae_dec_traj_np))
            all_file_name.extend(current_img)

            

    
    ADE_08 /= count
    ADE_12 /= count
    FDE_08 /= count
    FDE_12 /= count
    
    print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))

    return all_input,all_target,all_prediction,all_file_name

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
