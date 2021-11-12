import os
import json
import numpy as np
from .data_utils import bbox_denormalize, cxcywh_to_x1y1x2y2
from nuscenes.prediction import convert_local_coords_to_global
def compute_IOU(bbox_true, bbox_pred, format='xywh'):
    '''
    compute IOU
    [cx, cy, w, h] or [x1, y1, x2, y2]
    '''
    if format == 'xywh':
        xmin = np.max([bbox_true[0] - bbox_true[2]/2, bbox_pred[0] - bbox_pred[2]/2]) 
        xmax = np.min([bbox_true[0] + bbox_true[2]/2, bbox_pred[0] + bbox_pred[2]/2])
        ymin = np.max([bbox_true[1] - bbox_true[3]/2, bbox_pred[1] - bbox_pred[3]/2])
        ymax = np.min([bbox_true[1] + bbox_true[3]/2, bbox_pred[1] + bbox_pred[3]/2])
        w_true = bbox_true[2]
        h_true = bbox_true[3]
        w_pred = bbox_pred[2]
        h_pred = bbox_pred[3]
    elif format == 'x1y1x2y2':
        xmin = np.max([bbox_true[0], bbox_pred[0]])
        xmax = np.min([bbox_true[2], bbox_pred[2]])
        ymin = np.max([bbox_true[1], bbox_pred[1]])
        ymax = np.min([bbox_true[3], bbox_pred[3]])
        w_true = bbox_true[2] - bbox_true[0]
        h_true = bbox_true[3] - bbox_true[1]
        w_pred = bbox_pred[2] - bbox_pred[0]
        h_pred = bbox_pred[3] - bbox_pred[1]
    else:
        raise NameError("Unknown format {}".format(format))
    w_inter = np.max([0, xmax - xmin])
    h_inter = np.max([0, ymax - ymin])
    intersection = w_inter * h_inter
    union = (w_true * h_true + w_pred * h_pred) - intersection

    return intersection/union

def eval_jaad_pie(input_traj_np, target_traj_np, all_dec_traj_np):
    MSE_15=0
    MSE_05=0
    MSE_10=0
    FMSE=0
    CMSE=0
    CFMSE=0
    FIOU=0
    for batch_index in range(all_dec_traj_np.shape[0]):
        input_traj = np.expand_dims(input_traj_np[batch_index], axis=1)

        target_traj = input_traj + target_traj_np[batch_index]
        all_dec_traj = input_traj + all_dec_traj_np[batch_index]

        all_dec_traj = bbox_denormalize(all_dec_traj, W=1920, H=1080)
        target_traj = bbox_denormalize(target_traj, W=1920, H=1080)

        all_dec_traj_xyxy = cxcywh_to_x1y1x2y2(all_dec_traj)
        target_traj_xyxy = cxcywh_to_x1y1x2y2(target_traj)


        MSE_15 += np.square(target_traj_xyxy[-1,0:45,:] - all_dec_traj_xyxy[-1,0:45,:]).mean(axis=None)
        MSE_05 += np.square(target_traj_xyxy[-1,0:15,:] - all_dec_traj_xyxy[-1,0:15,:]).mean(axis=None)
        MSE_10 += np.square(target_traj_xyxy[-1,0:30,:] - all_dec_traj_xyxy[-1,0:30,:]).mean(axis=None)

        FMSE +=np.square(target_traj_xyxy[-1,44,:] - all_dec_traj_xyxy[-1,44,:]).mean(axis=None)


        CMSE += np.square(target_traj[-1,0:45,:2] - all_dec_traj[-1,0:45,:2]).mean(axis=None)
        CFMSE += np.square(target_traj[-1,44,:2] - all_dec_traj[-1,44,:2]).mean(axis=None)
        tmp_FIOU = []
        for i in range(target_traj_xyxy.shape[0]):
            tmp_FIOU.append(compute_IOU(target_traj_xyxy[i,44,:], all_dec_traj_xyxy[i,44,:], format='x1y1x2y2'))
        FIOU += np.mean(tmp_FIOU)
    return MSE_15, MSE_05, MSE_10, FMSE, CMSE, CFMSE, FIOU


def eval_jaad_pie_cvae(input_traj, target_traj, cvae_all_dec_traj):
    MSE_15=0
    MSE_05=0
    MSE_10=0
    FMSE=0
    CMSE=0
    CFMSE=0
    FIOU=0
    K = cvae_all_dec_traj.shape[2]
    tiled_target_traj = np.tile(target_traj[:, :, None, :], (1, 1, K, 1))
    #import pdb; pdb.set_trace()
    input_traj = np.tile(input_traj[:,-1,:][:,None, None,:], (1, 1, K, 1))
    #import pdb; pdb.set_trace()
    tiled_target_traj += input_traj
    cvae_all_dec_traj += input_traj
    
    tiled_target_traj = bbox_denormalize(tiled_target_traj, W=1920, H=1080)
    cvae_all_dec_traj = bbox_denormalize(cvae_all_dec_traj, W=1920, H=1080)

    tiled_target_traj_xyxy = cxcywh_to_x1y1x2y2(tiled_target_traj)
    cvae_all_dec_traj_xyxy = cxcywh_to_x1y1x2y2(cvae_all_dec_traj)

    MSE_05 = np.square(cvae_all_dec_traj_xyxy[:,:15,:,:] - tiled_target_traj_xyxy[:,:15,:,:]).mean(axis=(1, 3)).min(axis=-1).sum()
    #import pdb; pdb.set_trace()
    MSE_10 = np.square(cvae_all_dec_traj_xyxy[:,:30,:,:] - tiled_target_traj_xyxy[:,:30,:,:]).mean(axis=(1, 3)).min(axis=-1).sum()
    MSE_15 = np.square(cvae_all_dec_traj_xyxy - tiled_target_traj_xyxy).mean(axis=(1, 3)).min(axis=-1).sum()
    FMSE = np.square(cvae_all_dec_traj_xyxy[:,-1,:,:] - tiled_target_traj_xyxy[:,-1,:,:]).mean(axis=-1).min(axis=-1).sum()
    CMSE = np.square(cvae_all_dec_traj[:,:,:,:2] - tiled_target_traj[:,:,:,:2]).mean(axis=(1, 3)).min(axis=-1).sum()
    CFMSE = np.square(cvae_all_dec_traj[:,-1,:,:2] - tiled_target_traj[:,-1,:,:2]).mean(axis=-1).min(axis=-1).sum()
    return MSE_15, MSE_05, MSE_10, FMSE, CMSE, CFMSE, FIOU

def eval_hevi(input_traj_np, target_traj_np, all_dec_traj_np):
    ADE_15=0
    ADE_05=0
    ADE_10=0
    FDE=0
    CADE=0
    CFDE=0
    FIOU=0
    for batch_index in range(all_dec_traj_np.shape[0]):
        input_traj = np.expand_dims(input_traj_np[batch_index], axis=1)
        target_traj = input_traj + target_traj_np[batch_index]
        all_dec_traj = input_traj + all_dec_traj_np[batch_index]

        target_traj = bbox_denormalize(target_traj, W=1280, H=640)
        all_dec_traj = bbox_denormalize(all_dec_traj, W=1280, H=640)

        target_traj_xyxy = cxcywh_to_x1y1x2y2(target_traj)
        all_dec_traj_xyxy = cxcywh_to_x1y1x2y2(all_dec_traj)


        ADE_15 += np.mean(np.sqrt(np.sum((target_traj_xyxy[:,:,:2] - all_dec_traj_xyxy[:,:,:2]) ** 2, axis=-1)))
                    
        ADE_05 += np.mean(np.sqrt(np.sum((target_traj_xyxy[:,0:5,:2] - all_dec_traj_xyxy[:,0:5,:2]) ** 2, axis=-1)))
        ADE_10 += np.mean(np.sqrt(np.sum((target_traj_xyxy[:,0:10,:2] - all_dec_traj_xyxy[:,0:10,:2]) ** 2, axis=-1)))
        FDE += np.mean(np.sqrt(np.sum((target_traj_xyxy[:,-1,:2] - all_dec_traj_xyxy[:,-1,:2]) ** 2, axis=-1)))


        CADE += np.mean(np.sqrt(np.sum((target_traj[:,:,:2] - all_dec_traj[:,:,:2]) ** 2, axis=-1)))
        CFDE += np.mean(np.sqrt(np.sum((target_traj[:,-1,:2] - all_dec_traj[:,-1,:2]) ** 2, axis=-1)))
        tmp_FIOU = []
        for i in range(target_traj_xyxy.shape[0]):
            tmp_FIOU.append(compute_IOU(target_traj_xyxy[i,-1,:], all_dec_traj_xyxy[i,-1,:], format='x1y1x2y2'))
        FIOU += np.mean(tmp_FIOU)
    return ADE_15, ADE_05, ADE_10, FDE, CADE, CFDE, FIOU

def eval_ethucy(input_traj_np, target_traj_np, all_dec_traj_np):
    ADE_08=0
    ADE_12=0
    FDE_08=0
    FDE_12=0
    for batch in range(all_dec_traj_np.shape[0]):
        input_traj = np.expand_dims(input_traj_np[batch], axis=1)
        target_traj = input_traj[...,:2] + target_traj_np[batch]
        all_dec_traj = input_traj[...,:2] + all_dec_traj_np[batch]

        ADE_08 += np.mean(np.sqrt(np.sum((target_traj[-1,:8,:] - all_dec_traj[-1,:8,:]) ** 2, axis=-1)))
        ADE_12 += np.mean(np.sqrt(np.sum((target_traj[-1,:,:] - all_dec_traj[-1,:,:]) ** 2, axis=-1)))

        FDE_08 += np.mean(np.sqrt(np.sum((target_traj[-1,7,:] - all_dec_traj[-1,7,:]) ** 2, axis=-1)))
        FDE_12 += np.mean(np.sqrt(np.sum((target_traj[-1,-1,:] - all_dec_traj[-1,-1,:]) ** 2, axis=-1)))
    return ADE_08, FDE_08, ADE_12, FDE_12


def eval_ethucy_cvae(input_traj, target_traj, cvae_all_traj):
    result = {'ADE_08':0, 'ADE_12':0, 'FDE_08':0, 'FDE_12':0}

    K = cvae_all_traj.shape[2]
    tiled_target_traj = np.tile(target_traj[:, :, None, :], (1, 1, K, 1))
    #import pdb; pdb.set_trace()
    input_traj = np.tile(input_traj[:,-1,:][:,None, None,:], (1, 1, K, 1))

    result['ADE_08'] = np.linalg.norm(cvae_all_traj[:,:8,:,:] - tiled_target_traj[:,:8,:,:], axis=-1).mean(axis=1).min(axis=1).sum()
    result['ADE_12'] = np.linalg.norm(cvae_all_traj[:,:12,:,:] - tiled_target_traj[:,:12,:,:], axis=-1).mean(axis=1).min(axis=1).sum()
    result['FDE_08'] = np.linalg.norm(cvae_all_traj[:,7,:,:] - tiled_target_traj[:,7,:,:], axis=-1).min(axis=1).sum()
    result['FDE_12'] = np.linalg.norm(cvae_all_traj[:,11,:,:] - tiled_target_traj[:,11,:,:], axis=-1).min(axis=1).sum()


    return result

def eval_nuscenes_local(starting_translation, starting_rotation, target_traj, cvae_all_traj):
    result = {'ADE_12':0, 'FDE_12':0}


    K = cvae_all_traj.shape[2]
    B = cvae_all_traj.shape[0]
    tiled_target_traj = np.tile(target_traj[:, :, None, :], (1, 1, K, 1))
    
    cvae_all_traj_global = np.zeros(cvae_all_traj.shape)
    for k in range(K):
        for b in range(B):
            cvae_all_traj_global[b,:,k,:] = convert_local_coords_to_global(cvae_all_traj[b,:,k,:],starting_translation[b] ,starting_rotation[b]) 
    result['ADE_12'] = np.linalg.norm(cvae_all_traj_global[:,:12,:,:] - tiled_target_traj[:,:12,:,:], axis=-1).mean(axis=1).min(axis=1).sum()
    result['FDE_12'] = np.linalg.norm(cvae_all_traj_global[:,11,:,:] - tiled_target_traj[:,11,:,:], axis=-1).min(axis=1).sum()


    return result



def eval_nuscenes_api(starting_translation, starting_rotation, target_traj, cvae_all_traj, total_probabilities, tokens):
    result = {'ADE_12':0, 'FDE_12':0}


    K = cvae_all_traj.shape[2]
    B = cvae_all_traj.shape[0]
    tiled_target_traj = np.tile(target_traj[:, :, None, :], (1, 1, K, 1))
    preds5 = []
    cvae_all_traj_global = np.zeros(cvae_all_traj.shape)
    for k in range(K):
        for b in range(B):
            cvae_all_traj_global[b,:,k,:] = convert_local_coords_to_global(cvae_all_traj[b,:,k,:],starting_translation[b] ,starting_rotation[b])
    
    cvae_all_traj_global = np.transpose(cvae_all_traj_global, (0,2,1,3))

    tiled_target_traj = np.transpose(tiled_target_traj, (0,2,1,3))
    for i, token in enumerate(tokens):
        
        instance_token, sample_token = token.split("_")
        prediction = Prediction(instance=instance_token, sample=sample_token, prediction=cvae_all_traj_global[i],
                                        probabilities=total_probabilities[i]).serialize()
        preds5.append(prediction)

    return preds5