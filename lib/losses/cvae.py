import torch

def cvae_multi(pred_traj, target, first_history_index = 0):
        '''
        CVAE loss use best-of-many
        '''
        K = pred_traj.shape[3]
        
        target = target.unsqueeze(3).repeat(1, 1, 1, K, 1)
        total_loss = []
        for enc_step in range(first_history_index, pred_traj.size(1)):
            traj_rmse = torch.sqrt(torch.sum((pred_traj[:,enc_step,:,:,:] - target[:,enc_step,:,:,:])**2, dim=-1)).sum(dim=1)
            best_idx = torch.argmin(traj_rmse, dim=1)
            loss_traj = traj_rmse[range(len(best_idx)), best_idx].mean()
            total_loss.append(loss_traj)
        
        return sum(total_loss)/len(total_loss)
