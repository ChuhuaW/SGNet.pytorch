## Code modified based on https://github.com/MoonBlvd/bidireaction-trajectory-prediction/blob/main/datasets/JAAD.py

import numpy as np
import torch
from torch.utils import data
from .JAAD_origin import JAAD
from copy import deepcopy

class JAADDataLayer(data.Dataset):
    def __init__(self, args, split):
        self.split = split
        self.root = args.data_root
        self.args = args
        data_opts = {'fstride': 1,
                 'sample_type': 'all',
                 'height_rng': [0, float('inf')],
                 'squarify_ratio': 0,
                 'data_split_type': 'default',  # kfold, random, default
                 'seq_type': 'trajectory',
                 'min_track_size': 61,
                 'random_params': {'ratios': None,
                                 'val_data': True,
                                 'regen_data': True},
                 'kfold_params': {'num_folds': 5, 'fold': 1}}
        traj_model_opts = {'normalize_bbox': True,
                       'track_overlap': 0.5,
                       'observe_length': 15,
                       'predict_length': self.args.dec_steps,
                       'enc_input_type': ['bbox'],
                       'dec_input_type': [], #['intention_prob', 'obd_speed'],
                       'prediction_type': ['bbox'] 
                       }
        self.downsample_step = int(30/self.args.FPS)
        imdb = JAAD(data_path=self.root)
        #imdb.generate_database()
        beh_seq = imdb.generate_data_trajectory_sequence(self.split, **data_opts)
        self.data = self.get_data(beh_seq, **traj_model_opts)
    def __getitem__(self, index):
        obs_bbox = torch.FloatTensor(self.data['obs_bbox'][index])
        pred_bbox = torch.FloatTensor(self.data['pred_bbox'][index])
        gt_mean = torch.FloatTensor(self.data['gt_mean'][index])
        gt_std = torch.FloatTensor(self.data['gt_std'][index])
        cur_image_file = self.data['obs_image'][index][-1]
        ret = {'input_x':obs_bbox,
               'target_y':pred_bbox, 'cur_image_file':cur_image_file, 'gt_mean':gt_mean, 'gt_std':gt_std}
        ret['timestep'] = int(cur_image_file.split('/')[-1].split('.')[0])
        
        return ret

    def __len__(self):
        return len(self.data[list(self.data.keys())[0]])
        
    def get_tracks(self, dataset, data_types, observe_length, predict_length, overlap, normalize):
        """
        Generates tracks by sampling from pedestrian sequences
        :param dataset: The raw data passed to the method
        :param data_types: Specification of types of data for encoder and decoder. Data types depend on datasets. e.g.
        JAAD has 'bbox', 'ceneter' and PIE in addition has 'obd_speed', 'heading_angle', etc.
        :param observe_length: The length of the observation (i.e. time steps of the encoder)
        :param predict_length: The length of the prediction (i.e. time steps of the decoder)
        :param overlap: How much the sampled tracks should overlap. A value between [0,1) should be selected
        :param normalize: Whether to normalize center/bounding box coordinates, i.e. convert to velocities. NOTE: when
        the tracks are normalized, observation length becomes 1 step shorter, i.e. first step is removed.
        :return: A dictinary containing sampled tracks for each data modality
        """
        #  Calculates the overlap in terms of number of frames
        seq_length = observe_length + predict_length
        overlap_stride = observe_length if overlap == 0 else \
            int((1 - overlap) * observe_length)
        overlap_stride = 1 if overlap_stride < 1 else overlap_stride

        #  Check the validity of keys selected by user as data type
        d = {}
        for dt in data_types:
            try:
                d[dt] = dataset[dt]
            except:
                raise KeyError('Wrong data type is selected %s' % dt)
        d['image'] = dataset['image']
        d['pid'] = dataset['pid']
        d['resolution'] = dataset['resolution']
        d['flow'] = []

        #  Sample tracks from sequneces
        for k in d.keys():
            tracks = []
            for track in d[k]:
                tracks.extend([track[i:i + seq_length] for i in
                            range(0, len(track) - seq_length + 1, overlap_stride)])
            d[k] = tracks

        #  Normalize tracks using FOL paper method, 
        d['bbox'] = self.convert_normalize_bboxes(d['bbox'], d['resolution'], 
                                                  self.args.normalize, self.args.bbox_type)
        return d

    def convert_normalize_bboxes(self, all_bboxes, all_resolutions, normalize, bbox_type):
        '''input box type is x1y1x2y2 in original resolution'''
        for i in range(len(all_bboxes)):
            if len(all_bboxes[i]) == 0:
                continue
            bbox = np.array(all_bboxes[i])
            # NOTE ltrb to cxcywh
            if bbox_type == 'cxcywh':
                bbox[..., [2, 3]] = bbox[..., [2, 3]] - bbox[..., [0, 1]]
                bbox[..., [0, 1]] += bbox[..., [2, 3]]/2
            # NOTE Normalize bbox
            if normalize == 'zero-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.args.min_bbox)[None, :]
                _max = np.array(self.args.max_bbox)[None, :]
                bbox = (bbox - _min) / (_max - _min)
            elif normalize == 'plus-minus-one':
                # W, H  = all_resolutions[i][0]
                _min = np.array(self.args.min_bbox)[None, :]
                _max = np.array(self.args.max_bbox)[None, :]
                bbox = (2 * (bbox - _min) / (_max - _min)) - 1
            elif normalize == 'none':
                pass
            else:
                raise ValueError(normalize)
            all_bboxes[i] = bbox
        return all_bboxes

    def get_data_helper(self, data, data_type):
        """
        A helper function for data generation that combines different data types into a single representation
        :param data: A dictionary of different data types
        :param data_type: The data types defined for encoder and decoder input/output
        :return: A unified data representation as a list
        """
        if not data_type:
            return []
        d = []
        for dt in data_type:
            if dt == 'image':
                continue
            d.append(np.array(data[dt]))

        #  Concatenate different data points into a single representation
        if len(d) > 1:
            return np.concatenate(d, axis=2)
        elif len(d) == 1:
            return d[0]
        else:
            return d

    def get_data(self, data, **model_opts):
        """
        Main data generation function for training/testing
        :param data: The raw data
        :param model_opts: Control parameters for data generation characteristics (see below for default values)
        :return: A dictionary containing training and testing data
        """
        
        opts = {
            'normalize_bbox': True,
            'track_overlap': 0.5,
            'observe_length': 15,
            'predict_length': self.args.dec_steps,
            'enc_input_type': ['bbox'],
            'dec_input_type': [],
            'prediction_type': ['bbox']
        }
        for key, value in model_opts.items():
            assert key in opts.keys(), 'wrong data parameter %s' % key
            opts[key] = value

        observe_length = opts['observe_length']
        predict_length = opts['predict_length']
        data_types = set(opts['enc_input_type'] + opts['dec_input_type'] + opts['prediction_type'])
        data_tracks = self.get_tracks(data, data_types, observe_length,
                                      opts['predict_length'], opts['track_overlap'],
                                      opts['normalize_bbox'])

        obs_slices = {}
        pred_slices = {}
        obs_slices['gt_mean'] = []
        obs_slices['gt_std'] = []
        #  Generate observation/prediction sequences from the tracks
        for k in data_tracks.keys():

            obs_slices[k] = []
            pred_slices[k] = []
            # NOTE: Add downsample function
            down = self.downsample_step
            if k == 'bbox':
                start = down-1
                end = start + observe_length
                mean_list = []
                std_list = []
                observe_list = []
                target_list = []
                for sample in data_tracks[k]:
                    target = self.get_target(sample,start,end,observe_length,predict_length)
                    target_list.append(target)
                    observe = sample[down-1:observe_length:down]
                    observe_list.append(observe)
                    mean_np = np.zeros((observe_length, 4))
                    std_np = np.zeros((observe_length, 4))
                    for obs in range(1,observe_length+1):
                        whole_seq = sample[down-1:(obs + predict_length):down]
                        mean_np[obs-1] = deepcopy(whole_seq).mean(axis=0)
                        std_np[obs-1] = deepcopy(whole_seq).mean(axis=0)
                    mean_list.append(mean_np)
                    std_list.append(std_np)
                obs_slices[k].extend(observe_list)
                obs_slices['gt_mean'].extend(mean_list)
                obs_slices['gt_std'].extend(std_list)
                pred_slices[k].extend(target_list)

            else:
                obs_slices[k].extend([sample[down-1:observe_length:down] for sample in data_tracks[k]])
        ret =  {'obs_image': obs_slices['image'],
                'obs_pid': obs_slices['pid'],
                'obs_resolution': obs_slices['resolution'],
                'gt_mean': obs_slices['gt_mean'],
                'gt_std': obs_slices['gt_std'],
                'pred_image': pred_slices['image'],
                'pred_pid': pred_slices['pid'],
                'pred_resolution': pred_slices['resolution'],
                'obs_bbox': np.array(obs_slices['bbox']),
                'flow_input': obs_slices['flow'],
                'pred_bbox': np.array(pred_slices['bbox']), 
                'model_opts': opts,
                }
        
        return ret


    def get_target(self, session, start, end, observe_length, predict_length):
        '''
        Given the input session and the start and end time of the input clip, find the target
        TARGET FOR PREDICTION IS THE CHANGES IN THE FUTURE!!
        Params:
            session: the input time sequence of a car, can be bbox or ego_motion with shape (time, :)
            start: start frame id 
            end: end frame id
        Returns:
            target: Target tensor with shape (self.args.segment_len, dec_steps, :)
                    The target is the change of the values. e.g. target of yaw is \delta{\theta}_{t0,tn} 
        ''' 
        target = np.zeros((observe_length, predict_length, session.shape[-1]))
        for i, target_start in enumerate(range(start, end)):
            '''the target of time t is the change of bbox/ego motion at times [t+1,...,t+5}'''
            target_start = target_start + 1
            try:
                target[i,:,:] = np.asarray(session[target_start:target_start+predict_length,:] - 
                                           session[target_start-1:target_start,:])
            except:
                print("segment start: ", start)
                print("sample start: ", target_start)
                print("segment end: ", end)
                print(session.shape)
                raise ValueError()
        return target