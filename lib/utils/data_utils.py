import numpy as np
import random
import pickle as pkl
import os
import copy
import torch
import torch.utils.data as data
from lib.dataloaders import build_dataset

def set_seed(seed):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True



def build_data_loader(args, phase='train',batch_size=None):
    data_loaders = data.DataLoader(
        dataset=build_dataset(args, phase),
        batch_size=args.batch_size if batch_size is None else batch_size,
        shuffle=phase=='train',
        num_workers=args.num_workers,
        collate_fn=my_collate_fn if batch_size is not None else None)

    return data_loaders

def my_collate_fn(batch):
    return batch[0]

def cxcywh_to_x1y1x2y2(boxes):
    '''
    Params:
        boxes:(Cx, Cy, w, h)
    Returns:
        (x1, y1, x2, y2 or tlbr
    '''
    new_boxes = np.zeros_like(boxes)
    new_boxes[...,0] = boxes[...,0] - boxes[...,2]/2
    new_boxes[...,1] = boxes[...,1] - boxes[...,3]/2
    new_boxes[...,2] = boxes[...,0] + boxes[...,2]/2
    new_boxes[...,3] = boxes[...,1] + boxes[...,3]/2
    return new_boxes


def bbox_normalize(bbox,W=1280,H=640):
    '''
    normalize bbox value to [0,1]
    :Params:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
    :Return:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
    '''
    new_bbox = copy.deepcopy(bbox)
    new_bbox[:,0] /= W
    new_bbox[:,1] /= H
    new_bbox[:,2] /= W
    new_bbox[:,3] /= H
    
    return new_bbox

def bbox_denormalize(bbox,W=1280,H=640):
    '''
    normalize bbox value to [0,1]
    :Params:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to 1
    :Return:
        bbox: [cx, cy, w, h] with size (times, 4), value from 0 to W or H
    '''
    new_bbox = copy.deepcopy(bbox)
    new_bbox[..., 0] *= W
    new_bbox[..., 1] *= H
    new_bbox[..., 2] *= W
    new_bbox[..., 3] *= H
    
    return new_bbox


# FLow loading code adapted from:
# http://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy

def load_flow(flow_folder):
    '''
    Given video key, load the corresponding flow file
    '''
    flow_files = sorted(glob.glob(flow_folder + '*.flo'))
    flows = []
    for file in flow_files:
        flow = read_flo(file)
        flows.append(flow)
    return flows

TAG_FLOAT = 202021.25

def read_flo(file):
    assert type(file) is str, "file is not str %r" % str(file)
    assert os.path.isfile(file) is True, "file does not exist %r" % str(file)
    assert file[-4:] == '.flo', "file ending is not .flo %r" % file[-4:]
    f = open(file,'rb')
    flo_number = np.fromfile(f, np.float32, count=1)[0]
    assert flo_number == TAG_FLOAT, 'Flow number %r incorrect. Invalid .flo file' % flo_number
    w = int(np.fromfile(f, np.int32, count=1))
    h = int(np.fromfile(f, np.int32, count=1))
    #if error try: data = np.fromfile(f, np.float32, count=2*w[0]*h[0])
    data = np.fromfile(f, np.float32, count=2*w*h)
    # Reshape data into 3D array (columns, rows, bands)
    flow = np.resize(data, (int(h), int(w), 2))	
    f.close()

    return flow

