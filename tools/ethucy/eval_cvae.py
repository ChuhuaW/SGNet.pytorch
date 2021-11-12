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

import lib.utils as utl
from configs.ethucy import parse_sgd_args as parse_args
from lib.models import build_model
from lib.losses import rmse_loss
from lib.utils.ethucy_train_utils_cvae import train, val, test

def main(args):
    this_dir = osp.dirname(__file__)
    model_name = args.model
    save_dir = osp.join(this_dir, 'checkpoints', args.dataset,model_name,str(args.dropout), str(args.seed))
    if not osp.isdir(save_dir):
        os.makedirs(save_dir)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    utl.set_seed(int(args.seed))
    model = build_model(args)

    model = nn.DataParallel(model)
    model = model.to(device)
    if osp.isfile(args.checkpoint):

        checkpoint = torch.load(args.checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        del checkpoint

    criterion = rmse_loss().to(device)

    test_gen = utl.build_data_loader(args, 'test', batch_size = 1)
    print("Number of test samples:", test_gen.__len__())





    # test
    test_loss, ADE_08, FDE_08, ADE_12, FDE_12 = test(model, test_gen, criterion, device)
    print("Test Loss: {:.4f}".format(test_loss))
    print("ADE_08: %4f;  FDE_08: %4f;  ADE_12: %4f;   FDE_12: %4f\n" % (ADE_08, FDE_08, ADE_12, FDE_12))


        # # save checkpoints if loss decreases
        # if test_loss < min_loss:
        #     try:
        #         os.remove(best_model)
        #     except:
        #         pass

        #     min_loss = test_loss
        #     saved_model_name = 'epoch_' + str(format(epoch,'03')) + '_loss_%.4f'%min_loss + '.pth'

        #     print("Saving checkpoints: " + saved_model_name )
        #     if not os.path.isdir(save_dir):
        #         os.mkdir(save_dir)

        #     save_dict = {   'epoch': epoch,
        #                     'model_state_dict': model.state_dict(),
        #                     'optimizer_state_dict': optimizer.state_dict()}
        #     torch.save(save_dict, os.path.join(save_dir, saved_model_name))
        #     best_model = os.path.join(save_dir, saved_model_name)



        # if ADE_12 < min_ADE_12:
        #     try:
        #         os.remove(best_model_metric)
        #     except:
        #         pass
        #     min_ADE_08 = ADE_08
        #     min_FDE_08 = FDE_08
        #     min_ADE_12 = ADE_12
        #     min_FDE_12 = FDE_12
        #     with open(os.path.join(save_dir, 'metric.txt'),"w") as f:
        #         f.write("ADE_08: %4f; FDE_08: %4f; ADE_12: %4f; FDE_12: %4f;" % (ADE_08, FDE_08,ADE_12,FDE_12))

        #     saved_model_metric_name = 'metric_epoch_' + str(format(epoch,'03')) + '_loss_%.4f'%min_loss + '.pth'


        #     print("Saving checkpoints: " + saved_model_metric_name)
        #     if not os.path.isdir(save_dir):
        #         os.mkdir(save_dir)
        #     save_dict = {   'epoch': epoch,
        #                     'model_state_dict': model.state_dict(),
        #                     'optimizer_state_dict': optimizer.state_dict()}
        #     torch.save(save_dict, os.path.join(save_dir, saved_model_metric_name))


        #     best_model_metric = os.path.join(save_dir, saved_model_metric_name)

if __name__ == '__main__':
    main(parse_args())
