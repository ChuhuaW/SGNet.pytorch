import torch
import torch.nn as nn
from .feature_extractor import build_feature_extractor
from .bitrap_np import BiTraPNP
import torch.nn.functional as F

class SGNet_CVAE(nn.Module):
    def __init__(self, args):
        super(SGNet_CVAE, self).__init__()
        self.cvae = BiTraPNP(args)
        self.hidden_size = args.hidden_size # GRU hidden size
        self.enc_steps = args.enc_steps # observation step
        self.dec_steps = args.dec_steps # prediction step
        self.dataset = args.dataset
        self.dropout = args.dropout
        self.feature_extractor = build_feature_extractor(args)
        self.pred_dim = args.pred_dim
        self.K = args.K
        self.map = False
        if self.dataset in ['JAAD','PIE']:
            # the predict shift is in pixel
            self.pred_dim = 4
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                     self.pred_dim),
                                                     nn.Tanh())
            self.flow_enc_cell = nn.GRUCell(self.hidden_size*2, self.hidden_size)
        elif self.dataset in ['ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            self.pred_dim = 2
            # the predict shift is in meter
            self.regressor = nn.Sequential(nn.Linear(self.hidden_size, 
                                                        self.pred_dim))   
        self.enc_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))
        self.dec_goal_attn = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                1),
                                                nn.ReLU(inplace=True))

        self.enc_to_goal_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size//4),
                                                nn.ReLU(inplace=True))
        self.goal_hidden_to_traj = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.cvae_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size + args.LATENT_DIM,
                                                self.hidden_size),
                                                nn.ReLU(inplace=True))
        self.enc_to_dec_hidden = nn.Sequential(nn.Linear(self.hidden_size,
                                                self.hidden_size),
                                                nn.ReLU(inplace=True))

        self.goal_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.dec_hidden_to_input = nn.Sequential(nn.Linear(self.hidden_size,
                                                    self.hidden_size),
                                                    nn.ReLU(inplace=True))
        self.goal_to_enc = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.goal_to_dec = nn.Sequential(nn.Linear(self.hidden_size//4,
                                                    self.hidden_size//4),
                                                    nn.ReLU(inplace=True))
        self.enc_drop = nn.Dropout(self.dropout)
        self.goal_drop = nn.Dropout(self.dropout)
        self.dec_drop = nn.Dropout(self.dropout)
        self.traj_enc_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
        self.goal_cell = nn.GRUCell(self.hidden_size//4, self.hidden_size//4)
        self.dec_cell = nn.GRUCell(self.hidden_size + self.hidden_size//4, self.hidden_size)
    
    def SGE(self, goal_hidden):
        # initial goal input with zero
        goal_input = goal_hidden.new_zeros((goal_hidden.size(0), self.hidden_size//4))
        # initial trajectory tensor
        goal_traj = goal_hidden.new_zeros(goal_hidden.size(0), self.dec_steps, self.pred_dim)
        goal_list = []
        for dec_step in range(self.dec_steps):
            goal_hidden = self.goal_cell(self.goal_drop(goal_input), goal_hidden)
            # next step input is generate by hidden
            goal_input = self.goal_hidden_to_input(goal_hidden)
            goal_list.append(goal_hidden)
            # regress goal traj for loss
            goal_traj_hidden = self.goal_hidden_to_traj(goal_hidden)
            goal_traj[:,dec_step,:] = self.regressor(goal_traj_hidden)
        # get goal for decoder and encoder
        goal_for_dec = [self.goal_to_dec(goal) for goal in goal_list]
        goal_for_enc = torch.stack([self.goal_to_enc(goal) for goal in goal_list],dim = 1)
        enc_attn= self.enc_goal_attn(torch.tanh(goal_for_enc)).squeeze(-1)
        enc_attn = F.softmax(enc_attn, dim =1).unsqueeze(1)
        goal_for_enc  = torch.bmm(enc_attn, goal_for_enc).squeeze(1)
        return goal_for_dec, goal_for_enc, goal_traj

    def cvae_decoder(self, dec_hidden, goal_for_dec):
        batch_size = dec_hidden.size(0)
       
        K = dec_hidden.shape[1]
        dec_hidden = dec_hidden.view(-1, dec_hidden.shape[-1])
        dec_traj = dec_hidden.new_zeros(batch_size, self.dec_steps, K, self.pred_dim)
        for dec_step in range(self.dec_steps):
            # incremental goal for each time step
            goal_dec_input = dec_hidden.new_zeros(batch_size, self.dec_steps, self.hidden_size//4)
            goal_dec_input_temp = torch.stack(goal_for_dec[dec_step:],dim=1)
            goal_dec_input[:,dec_step:,:] = goal_dec_input_temp
            dec_attn= self.dec_goal_attn(torch.tanh(goal_dec_input)).squeeze(-1)
            dec_attn = F.softmax(dec_attn, dim =1).unsqueeze(1)
            goal_dec_input  = torch.bmm(dec_attn,goal_dec_input).squeeze(1)
            goal_dec_input = goal_dec_input.unsqueeze(1).repeat(1, K, 1).view(-1, goal_dec_input.shape[-1])
            dec_dec_input = self.dec_hidden_to_input(dec_hidden)
            dec_input = self.dec_drop(torch.cat((goal_dec_input,dec_dec_input),dim = -1))
            dec_hidden = self.dec_cell(dec_input, dec_hidden)
            # regress dec traj for loss
            batch_traj = self.regressor(dec_hidden)
            batch_traj = batch_traj.view(-1, K, batch_traj.shape[-1])
            dec_traj[:,dec_step,:,:] = batch_traj
        return dec_traj

    def encoder(self, raw_inputs, raw_targets, traj_input, flow_input=None, start_index = 0):
        # initial output tensor
        all_goal_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.pred_dim)
        all_cvae_dec_traj = traj_input.new_zeros(traj_input.size(0), self.enc_steps, self.dec_steps, self.K, self.pred_dim)
        # initial encoder goal with zeros
        goal_for_enc = traj_input.new_zeros((traj_input.size(0), self.hidden_size//4))
        # initial encoder hidden with zeros
        traj_enc_hidden = traj_input.new_zeros((traj_input.size(0), self.hidden_size))
        total_probabilities = traj_input.new_zeros((traj_input.size(0), self.enc_steps, self.K))
        total_KLD = 0
        for enc_step in range(start_index, self.enc_steps):
            traj_enc_hidden = self.traj_enc_cell(self.enc_drop(torch.cat((traj_input[:,enc_step,:], goal_for_enc), 1)), traj_enc_hidden)
            enc_hidden = traj_enc_hidden
            goal_hidden = self.enc_to_goal_hidden(enc_hidden)
            goal_for_dec, goal_for_enc, goal_traj = self.SGE(goal_hidden)
            all_goal_traj[:,enc_step,:,:] = goal_traj
            dec_hidden = self.enc_to_dec_hidden(enc_hidden)
            if self.training:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:,enc_step,:], self.K, raw_targets[:,enc_step,:,:])
            else:
                cvae_hidden, KLD, probability = self.cvae(dec_hidden, raw_inputs[:,enc_step,:], self.K)
            total_probabilities[:,enc_step,:] = probability
            total_KLD += KLD
            cvae_dec_hidden= self.cvae_to_dec_hidden(cvae_hidden)
            if self.map:
                map_input = flow_input
                cvae_dec_hidden = (cvae_dec_hidden + map_input.unsqueeze(1))/2
            all_cvae_dec_traj[:,enc_step,:,:,:] = self.cvae_decoder(cvae_dec_hidden, goal_for_dec)
        return all_goal_traj, all_cvae_dec_traj, total_KLD, total_probabilities
            
    def forward(self, inputs, map_mask=None, targets = None, start_index = 0, training=True):
        self.training = training
        if torch.is_tensor(start_index):
            start_index = start_index[0].item()
        if self.dataset in ['JAAD','PIE']:
            traj_input = self.feature_extractor(inputs)
            all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input)
            return all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities
        elif self.dataset in ['ETH', 'HOTEL','UNIV','ZARA1', 'ZARA2']:
            traj_input_temp = self.feature_extractor(inputs[:,start_index:,:])
            traj_input = traj_input_temp.new_zeros((inputs.size(0), inputs.size(1), traj_input_temp.size(-1)))
            traj_input[:,start_index:,:] = traj_input_temp
            all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities = self.encoder(inputs, targets, traj_input, None, start_index)
            return all_goal_traj, all_cvae_dec_traj, KLD, total_probabilities