import torch
import torch.nn as nn


class rmse_loss(nn.Module):
    '''
    Params:
        x_pred: (batch_size, enc_steps, dec_steps, pred_dim)
        x_true: (batch_size, enc_steps, dec_steps, pred_dim)
    Returns:
        rmse: scalar, rmse = \sum_{i=1:batch_size}()
    '''
    def __init__(self):
        super(rmse_loss, self).__init__()
    
    def forward(self, x_pred, x_true):
        L2_diff = torch.sqrt(torch.sum((x_pred - x_true)**2, dim=3))
        # sum over prediction time steps
        L2_all_pred = torch.sum(L2_diff, dim=2)
        # mean of each frames predictions
        L2_mean_pred = torch.mean(L2_all_pred, dim=1)
        # sum of all batches
        L2_mean_pred = torch.mean(L2_mean_pred, dim=0)
        return L2_mean_pred