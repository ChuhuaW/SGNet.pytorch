'''
Defined classes:
    class BiTraPNP()
Some utilities are cited from Trajectron++
'''
import sys
import numpy as np
import copy
from collections import defaultdict
import torch
from torch import nn, optim
from torch.nn import functional as F
import torch.nn.utils.rnn as rnn
from torch.distributions import Normal

def reconstructed_probability(x):
    recon_dist = Normal(0, 1)
    p = recon_dist.log_prob(x).exp().mean(dim=-1)  # [batch_size, K]
    return p

class BiTraPNP(nn.Module):
    def __init__(self, args):
        super(BiTraPNP, self).__init__()
        self.args = copy.deepcopy(args)
        self.param_scheduler = None
        self.input_dim = self.args.input_dim
        self.pred_dim = self.args.pred_dim
        self.hidden_size = self.args.hidden_size
        self.nu = args.nu
        self.sigma = args.sigma
        self.node_future_encoder_h = nn.Sequential(nn.Linear(self.input_dim, self.hidden_size//2),nn.ReLU())
        self.gt_goal_encoder = nn.GRU(input_size=self.pred_dim,
                                        hidden_size=self.hidden_size//2,
                                        bidirectional=True,
                                        batch_first=True)
        self.p_z_x = nn.Sequential(nn.Linear(self.hidden_size,  
                                            128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.args.LATENT_DIM*2))
        # posterior
        self.q_z_xy = nn.Sequential(nn.Linear(self.hidden_size + self.hidden_size,
                                            128),
                                    nn.ReLU(),
                                    nn.Linear(128, 64),
                                    nn.ReLU(),
                                    nn.Linear(64, self.args.LATENT_DIM*2))
        
        

    def gaussian_latent_net(self, enc_h, cur_state, K,  target=None, z_mode=None):
        # get mu, sigma
        # 1. sample z from piror
        z_mu_logvar_p = self.p_z_x(enc_h)
        z_mu_p = z_mu_logvar_p[:, :self.args.LATENT_DIM]
        z_logvar_p = z_mu_logvar_p[:, self.args.LATENT_DIM:]
        if target is not None:
            # 2. sample z from posterior, for training only
            initial_h = self.node_future_encoder_h(cur_state)
            initial_h = torch.stack([initial_h, torch.zeros_like(initial_h, device=initial_h.device)], dim=0)
            self.gt_goal_encoder.flatten_parameters()
            _, target_h = self.gt_goal_encoder(target, initial_h)
            target_h = target_h.permute(1,0,2)
            target_h = target_h.reshape(-1, target_h.shape[1] * target_h.shape[2])
            
            z_mu_logvar_q = self.q_z_xy(torch.cat([enc_h, target_h], dim=-1))
            z_mu_q = z_mu_logvar_q[:, :self.args.LATENT_DIM]
            z_logvar_q = z_mu_logvar_q[:, self.args.LATENT_DIM:]
            Z_mu = z_mu_q
            Z_logvar = z_logvar_q

            # 3. compute KL(q_z_xy||p_z_x)
            KLD = 0.5 * ((z_logvar_q.exp()/z_logvar_p.exp()) + \
                        (z_mu_p - z_mu_q).pow(2)/z_logvar_p.exp() - \
                        1 + \
                        (z_logvar_p - z_logvar_q))
            KLD = KLD.sum(dim=-1).mean()
            KLD = torch.clamp(KLD, min=0.001)
            
        else:
            Z_mu = z_mu_p
            Z_logvar = z_logvar_p
            KLD = torch.as_tensor(0.0, device=Z_logvar.device)
        
        # 4. Draw sample
        with torch.set_grad_enabled(False):
            K_samples = torch.normal(self.nu, self.sigma, size = (enc_h.shape[0], K, self.args.LATENT_DIM)).cuda()

        probability = reconstructed_probability(K_samples)
        Z_std = torch.exp(0.5 * Z_logvar)
        Z = Z_mu.unsqueeze(1).repeat(1, K, 1) + K_samples * Z_std.unsqueeze(1).repeat(1, K, 1)
        if z_mode:
            Z = torch.cat((Z_mu.unsqueeze(1), Z), dim=1)

        
        return Z, KLD, probability


    def forward(self, h_x, last_input, K, target_y=None):
        '''
        Params:

        '''
        Z, KLD, probability = self.gaussian_latent_net(h_x, last_input, K, target_y, z_mode=False)
        enc_h_and_z = torch.cat([h_x.unsqueeze(1).repeat(1, Z.shape[1], 1), Z], dim=-1)
        dec_h = enc_h_and_z if self.args.DEC_WITH_Z else h_x
        return dec_h, KLD, probability
