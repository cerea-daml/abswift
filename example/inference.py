#Sample train, with physicsnemo for optimisations and hydra for configuration management

#TODO: messed up imports
import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import os

import torch

from physicsnemo.distributed import DistributedManager
from physicsnemo.launch.utils import load_checkpoint

from physicsnemo.launch.logging import PythonLogger
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper

from physicsnemo.models.meta import ModelMetaData
from dataclasses import dataclass
from physicsnemo.models.module import Module

from abswift.dataset.random_buildings_dataset import InMemoryRandomBuildingsDataset, denormalise
from abswift.model.abswift import ABSWIFT as ABSWIFT_torch
from abswift.model.abswift_collator import ABSWIFTCollator

import numpy as np
import torch

# import quickview.fonctions as qf #Don't want this dependency

import matplotlib.pyplot as plt
from matplotlib import patches

import cmocean

#configure pyplot
SMALL_SIZE = 13
MEDIUM_SIZE = 15
BIGGER_SIZE = 17

plt.rc('font', size=SMALL_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=SMALL_SIZE)     # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=SMALL_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title
plt.rcParams['mathtext.fontset'] = 'stix'
plt.rcParams['font.family'] = 'STIXGeneral'


#savefig params
plt.rcParams['savefig.dpi'] = 300 #dpi when saving
plt.rcParams['savefig.bbox'] = 'tight' #bbox when saving
plt.rcParams['savefig.format'] = 'png'

from abswift.dataset.vtk_tools import *

def plot_a_case(mesh, gt, pred, height = 2):

    gt = gt.copy()
    pred = pred.copy()

    for d in (gt, pred):
        d['k'] = 10 ** d.pop('log-k')
        d['epsilon'] = 10 ** d.pop('log-epsilon')

    err = {k:(gt[k] - pred[k])**2 / np.var(gt[k]) * 100 for k in gt.keys()}

    for d in (gt, pred, err):
        d['velocity'] = np.linalg.norm(d['velocity'], axis = -1)

    gt = {k:v.squeeze() for k,v in gt.items()}
    pred = {k:v.squeeze() for k,v in pred.items()}
    err = {k:v.squeeze() for k,v in err.items()}

    #apply slice
    triang, ids = get_slice(mesh, height)
    gt = {k:v[ids] for k,v in gt.items()}
    pred = {k:v[ids] for k,v in pred.items()}
    err = {k:v[ids] for k,v in err.items()}


    fig, allaxs = plt.subplots(5,3, 
                                   figsize = (12,8),
                                  sharex=True, sharey=True,
                                  layout = 'constrained', 
                                  subplot_kw = dict(aspect='equal'))
    axs1, axs2, axs3 = allaxs.T
    
    #plotting parameters for all fields
    all_kwargs = dict()
    allv = np.concatenate([gt['velocity'], pred['velocity']], axis = 0)
    all_kwargs['velocity'] = dict(cmap = cmocean.cm.dense, vmin = np.nanmin(allv), vmax = np.nanmax(allv))

    allk = np.concatenate([gt['k'], pred['k']], axis = 0)
    all_kwargs['k'] = dict(cmap = cmocean.cm.speed, vmin = np.nanmin(allk), vmax = np.nanmax(allk))

    alleps = np.concatenate([gt['epsilon'], pred['epsilon']], axis = 0)
    all_kwargs['epsilon'] = dict(cmap = cmocean.cm.amp, vmin = np.nanmin(alleps), vmax = np.nanmax(alleps))

    allp = np.concatenate([gt['pressure'], pred['pressure']], axis = 0)
    absmax = np.max([- np.nanmin(allp), np.nanmax(allp)])
    all_kwargs['pressure'] = {'vmin':-absmax, 'vmax':absmax, 'cmap':plt.get_cmap('bwr')}

    alltheta = np.concatenate([gt['pottemp'], pred['pottemp']], axis = 0)
    all_kwargs['pottemp'] = dict(cmap = cmocean.cm.thermal, vmin = np.nanmin(alltheta), vmax = np.nanmax(alltheta))
    
    labels = {
        'velocity': 'Velocity',
        'k': 'Turbulent\nkinetic\nenergy',
        'epsilon': 'Turbulent\nkinetic energy\ndissipation rate',
        'pressure': 'Pressure',
        'pottemp': 'Potential\ntemperature'
    }
    units = {
        'velocity': r'm $\cdot$ s$^{-1}$',
        'k': r'm²$\cdot$ s$^{-2}$',
        'epsilon': r'm²$\cdot$ s$^{-3}$',
        'pressure': r'Pa',
        'pottemp': r'°K'
    }

    #Make all buildings black
    for ax in allaxs.flatten():
        ax.add_patch(patches.Rectangle((-50, -50), 100, 100, color = 'teal'))
        
    #plot all
    for i,field in enumerate(gt.keys()):
        axs1[i].tripcolor(triang, gt[field], **all_kwargs.get(field, {}))
        tr_f = axs2[i].tripcolor(triang, pred[field], **all_kwargs.get(field, {}))

        # vmin = -max(err[field]) if field == 'apressure' else None #center pressure err
        # tr_err = axs3[i].tripcolor(triang, err[field], cmap = all_kwargs[field]['cmap'], vmin = vmin)
        tr_err = axs3[i].tripcolor(triang, err[field], cmap = cmocean.cm.gray_r,)

        fig.colorbar(tr_f, ax = [axs1[i], axs2[i]], shrink = 0.7, label = units.get(field, None))
        fig.colorbar(tr_err, ax = axs3[i], shrink = 0.7)
        axs1[i].set_ylabel(labels.get(field, None))

    axs1[0].set_title('CFD')
    axs2[0].set_title('AB-SWIFT')
    axs3[0].set_title('NMSE\n(% of variance)')

    # for ax in axs3:
    ticks = axs3[0].get_xticklabels()
    ticks = [t.get_text() + ' m' for t in ticks]
    axs3[0].set_xticklabels(ticks)
    
    # for ax in (axs1[0], axs2[0], axs3[0]):
    ticks = axs3[0].get_yticklabels()
    ticks = [t.get_text() + ' m' for t in ticks]
    axs3[0].set_yticklabels(ticks)

    fig.suptitle(f'Prediction at height {height}m')

    return fig

#Defining a Nvidia-physicsnemo model
@dataclass
class MetaData(ModelMetaData):
    name: str = "Anchored_Branched_Steady_state_WInd_Flow_Transformer"
    # Optimization
    jit: bool = False
    cuda_graphs: bool = False
    amp: bool = True

class ABSWIFT(Module):
    def __init__(
        self,
        *args, **kwargs
    ) -> None:
        super().__init__(meta=MetaData())
        self.model = ABSWIFT_torch(
            *args, **kwargs
        )
    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)
    


@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig) -> None:

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    #initialise model
    model = ABSWIFT(**cfg.model).eval()
    weights_path = to_absolute_path(cfg.inference.weights_path)
    epoch = load_checkpoint(weights_path, models = model)
    rank_zero_logger.info(f"Loaded model weights of epoch {epoch}")

    #initialise dataset
    cfg.dataset.data_dir = to_absolute_path(cfg.dataset.data_dir)
    dataset = InMemoryRandomBuildingsDataset(**cfg.dataset, split = 'test', normalise = True)
    collator = ABSWIFTCollator(**cfg.collator)

    #get the data sample
    rank_zero_logger.info(f"Running inference for test case {cfg.inference.idx}")
    normalised_data = dataset[cfg.inference.idx]

    #apply collator
    batch = collator([normalised_data])

    #Add all points as query points to simplify postprocessing
    pos = normalised_data['volume_position']
    batch['volume_query_position'] = pos.unsqueeze(0)
    
    #pop labels from batched data
    for field in cfg.dataset.volume_output_fields:
        batch.pop(f'volume_anchor_{field}', None)
        batch.pop(f'volume_query_{field}',  None)

    #send to device
    batch = {k:v.to(dist.device) for k,v in batch.items()}
    model = model.to(dist.device)

    #inference
    with torch.no_grad():
        out = model(**batch)
    out = {k:v.cpu() for k,v in out.items()}

    #postprocess
    pred, gt = {}, {}
    for k in cfg.dataset.volume_output_fields:
        pred[k] = denormalise(out[f'volume_query_{k}'][0], dataset.normalisation_stats[f'volume_{k}_mean'], dataset.normalisation_stats[f'volume_{k}_std'])
        gt[k] = denormalise(normalised_data[f'volume_{k}'], dataset.normalisation_stats[f'volume_{k}_mean'], dataset.normalisation_stats[f'volume_{k}_std'])

    gt = {k:v.numpy() for k,v in gt.items()}
    pred = {k:v.numpy() for k,v in pred.items()}

    #plot
    fig = plot_a_case(dataset.get_volume_mesh(cfg.inference.idx), gt, pred, height = cfg.inference.plot_height)
    fig.savefig(f'test_case_{cfg.inference.idx}')
    rank_zero_logger.info(f'Saved plot result test_case_{cfg.inference.idx}.png')

    return


if __name__ == "__main__":
    # setup comunication port
    # We define a port based on slurm job id
    # to have different port between different trainings submitted at the same time
    if 'SLURM_JOB_ID' in os.environ:
        os.environ['MASTER_PORT'] = str(1000+int(os.environ['SLURM_JOB_ID'])%1000)
        
    main()