#!/usr/bin/env python

'''Torch dataset and pipeline from the random buildings dataset'''

import os
import json

import numpy as np

from sklearn.preprocessing import MultiLabelBinarizer

# import quickview.fonctions as qf
from .vtk_tools import *
from .mo_profiles import compute_meteo_profile

import torch
from torch.utils.data import Dataset

### utility functions ###
_EPSILON = 1e-10

def normalise(data, mean, std):
    std = torch.clamp(std, min = _EPSILON)
    return (data - mean) / std

def denormalise(data, mean, std):
    return (data * std) + mean

### Dataset classes ###

class InMemoryRandomBuildingsDataset(Dataset):
    '''
    Dataset of wind flow around randomly placed buildings'''
    def __init__(self, 
                 data_dir, 
                 split,
                 volume_output_fields,
                 volume_feature = None,
                 buildings_feature = None,
                 ground_feature = None,
                 normalise = False,
                 position_scale : float|None = None):

        super().__init__()
        self._raw_dir = data_dir
        self._data = dict()

        #parse info
        with open(self._raw_dir+os.sep+'random_buildings_dataset.json', 'r') as f:
            info = json.load(f)

        #train test split
        self.split = split
        assert split in [None, 'all', 'test', 'train', 'val'], f'Split {split} is not implemented'
        if split is None or split == 'all':
            self._file_ids = np.arange(info['total_size'], dtype = int)
        elif split == 'train':
            self._file_ids = info['train_ids']
        elif split == 'test':
            self._file_ids = info['test_ids']
        elif split == 'val':
            self._file_ids = info['val_ids']

        node_type_classes = info['node_type_classes']
        self.node_type_transf = MultiLabelBinarizer(classes = node_type_classes).fit(None)

        self._invlmos = info['invlmo']
        self._z0s = info['z0']

        self._fields = [f for f in volume_output_fields if 'log' not in f]
        self._log_fields = [f for f in volume_output_fields if 'log' in f]

        self._volume_feature = volume_feature
        self._buildings_feature = buildings_feature
        self._ground_feature = ground_feature

        #load everything in memory
        for idx in range(len(self)):
            self._data[idx] = self._load_and_preprocess(idx)

        #setup normalisation
        if normalise:
            normalisation_stats = build_normalisation_stats_dict(volume_output_fields, volume_feature, buildings_feature, ground_feature)
            self.normalisation_stats = {k:torch.tensor(v) for k,v in normalisation_stats.items()}
            if position_scale is not None:
                self.normalise_position = True
                #these dimensions are sligltly wider than in reality to avoid errors
                self._pos_min = torch.tensor([-150, -120, 0])
                self._pos_max = torch.tensor([310, 120, 70])
                self._pos_scale = position_scale
            else:
                self.normalise_position = False

        self.normalise = normalise

    def _load_and_preprocess(self, idx):
        file_id = self._file_ids[idx]

        data = {}

        ####### PROCESS PROFILE #########
        invlmo = self._invlmos[file_id]
        z0 = self._z0s[file_id]

        lmo = 1 / invlmo if invlmo !=0 else 1e50
        z = np.linspace(0, 100, 64)
        profile = compute_meteo_profile(lmo = lmo, z0 = z0, zref=80.0, uref=6.0, t0=293.15, z = z)
        data['profile'] = profile

        ####### PROCESS VOLUME #######
        vtk_volume = read_vtk(f'{self._raw_dir}/volume_{file_id}.vtk')
        data['volume_position'] = get_coords(vtk_volume, loc = 'cells')

        #output fields
        fields = {f'volume_{f}':get_field(vtk_volume, f, loc = 'cells') for f in self._fields}
        log_fields = {f'volume_{f}':np.log10(get_field(vtk_volume, f.split('-')[1], loc = 'cells')) for f in self._log_fields}
        fields = {**fields, **log_fields}
        fields = {k:v[:,None] if len(v.shape)==1 else v  for k,v in fields.items()} #add extra dim for scalar fields
        data = {**data, **fields}

        #extra features
        if self._volume_feature is not None:
            n_pts = data['volume_position'].shape[0]
            volume_features = []
            for f in self._volume_feature:
                if f == 'invlmo':
                    volume_features.append(np.full((n_pts, 1), invlmo))
                elif f == 'z0':
                    volume_features.append(np.full((n_pts, 1), z0))
                else:
                    feat = get_field(vtk_volume, f, loc = 'cells')
                    feat = feat[:,None] if len(feat.shape) == 1 else feat
                    volume_features.append(feat)
            data['volume_feature'] = np.concatenate(volume_features, axis = 1)

        ########### PROCESS BUILDINGS ##########
        vtk_buildings = read_vtk(f'{self._raw_dir}/buildings_{file_id}.vtk')
        data['buildings_position'] = get_coords(vtk_buildings, loc = 'points')

        #extra features
        if self._buildings_feature is not None:
            n_pts = data['buildings_position'].shape[0]
            buildings_features = []
            for f in self._buildings_feature:
                if f == 'invlmo':
                    buildings_features.append(np.full((n_pts, 1), invlmo))
                elif f == 'z0':
                    buildings_features.append(np.full((n_pts, 1), z0))
                else:
                    feat = get_field(vtk_volume, f, loc = 'points')
                    feat = feat[:,None] if len(feat.shape) == 1 else feat
                    buildings_features.append(feat)
            data['buildings_feature'] = np.concatenate(buildings_features, axis = 1)

        ########## PROCESS GROUND ############
        vtk_ground = read_vtk(f'{self._raw_dir}/ground_{file_id}.vtk')
        data['ground_position'] = get_coords(vtk_ground, loc = 'points')

        #extra features
        if self._ground_feature is not None:
            n_pts = data['ground_position'].shape[0]
            ground_features = []
            for f in self._ground_feature:
                if f == 'invlmo':
                    ground_features.append(np.full((n_pts, 1), invlmo))
                elif f == 'z0':
                    ground_features.append(np.full((n_pts, 1), z0))
                else:
                    feat = get_field(vtk_volume, f, loc = 'points')
                    feat = feat[:,None] if len(feat.shape) == 1 else feat
                    ground_features.append(feat)
            data['ground_feature'] = np.concatenate(ground_features, axis = 1)

        #tensorify
        data = {k:torch.tensor(v, dtype=torch.float32) for k,v in data.items()}

        return data

    
    def __getitem__(self, idx):

        data = self._data[idx]

        #normalisation is done here to avoid modyfing raw data
        if self.normalise:
            normalised_data = {k:normalise(data[k], self.normalisation_stats[k+'_mean'], self.normalisation_stats[k+'_std']) 
                               for k in data.keys() if 'position' not in k}
            #position normalisation
            if self.normalise_position:
                normalised_positional_data = {k:(v-self._pos_min) / (self._pos_max - self._pos_min) * self._pos_scale
                                              for k,v in data.items() if 'position' in k}
            else:
                normalised_positional_data = {k:v for k,v in data.items() if 'position' in k}
            data = {**normalised_data, **normalised_positional_data}
        
        return data
        
    def __len__(self):
        return len(self._file_ids)
    
    def get_volume_mesh(self, idx):
        '''Return the vtk volume mesh (for plots)'''
        file_id = self._file_ids[idx]
        vtk_volume = read_vtk(f'{self._raw_dir}/volume_{file_id}.vtk')
        return vtk_volume
    

#normalisation stats (05/01/26 computation, updated from extra data)
_fields_stats = dict(
    velocity = ([2.8041477e+00, -8.3266100e-04,  8.5222470e-03], [1.7015716,   0.18931563,  0.12401797]),
    epsilon = ([3.1351170e-01], [0.91677713]),
    k = ([2.4578676e+00], [4.480781]),
    pottemp = ([2.6771170e+02], [20.573137]),
    pressure = ([-9.3729353e-01], [7.769606]),
)
_fields_stats['log-k'] = ([-4.7141120e-01], [0.99278253])
_fields_stats['log-epsilon'] = ([-1.7001820e+00], [1.1928372])

_profile_stats = ([4.9893808e+00, 2.8492166e+02, 5.7216582e+00, 9.3904741e-02], [ 1.3752371,  14.820417,   10.746456,    0.37050286])
_features_stats = dict(
    invlmo = ([0.0015681160508819359], [0.09660768788320781]),
    z0 = ([0.3117240183710031], [0.2536357233089424]),
    node_type = ([0,0,0,0,0,0,0,0], [1,1,1,1,1,1,1,1]), #unormalised because we already have 0/1 categorial data
    normals = ([0,0,0], [1,1,1]) #unormalised because normals already have ||.|| = 1
)

def build_normalisation_stats_dict(volume_fields: list[str], volume_feature = None, buildings_feature = None, ground_feature = None):
    '''Build a dictionary of normalisation stats corresponding to the given fields'''

    normalisation_stats = dict(
        profile_mean = _profile_stats[0],
        profile_std  = _profile_stats[1],
        **{f'volume_{field}_mean':_fields_stats[field][0] for field in volume_fields},
        **{f'volume_{field}_std': _fields_stats[field][1] for field in volume_fields},
    )
    if volume_feature is not None:
        normalisation_stats['volume_feature_mean'] = np.concatenate([_features_stats[f][0] for f in volume_feature]),
        normalisation_stats['volume_feature_std']  = np.concatenate([_features_stats[f][1] for f in volume_feature]),
    if buildings_feature is not None:
        normalisation_stats['buildings_feature_mean'] = np.concatenate([_features_stats[f][0] for f in buildings_feature]),
        normalisation_stats['buildings_feature_std']  = np.concatenate([_features_stats[f][1] for f in buildings_feature]),
    if ground_feature is not None:
        normalisation_stats['ground_feature_mean'] = np.concatenate([_features_stats[f][0] for f in ground_feature]),
        normalisation_stats['ground_feature_std']  = np.concatenate([_features_stats[f][1] for f in ground_feature]),
    
    #cast to float32 so that tensors don't get cast to double during normalisation
    normalisation_stats = {k:np.array(v, dtype = np.float32) for k,v in normalisation_stats.items()}
    return normalisation_stats