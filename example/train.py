#Sample train, with physicsnemo for optimisations and hydra for configuration management

import hydra
from omegaconf import DictConfig
from hydra.utils import to_absolute_path

import os, time

import torch
from torch.optim import lr_scheduler, AdamW
from torch.nn.parallel import DistributedDataParallel

from physicsnemo.distributed import DistributedManager
from physicsnemo.utils import StaticCaptureTraining, StaticCaptureEvaluateNoGrad
from physicsnemo.launch.utils import load_checkpoint, save_checkpoint

from physicsnemo.launch.logging import PythonLogger, LaunchLogger
from physicsnemo.launch.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.profiling import profile, Profiler

from physicsnemo.models.meta import ModelMetaData
from dataclasses import dataclass
from physicsnemo.models.module import Module

from omegaconf import OmegaConf

from abswift.dataset.random_buildings_dataset import InMemoryRandomBuildingsDataset
from abswift.model.abswift import ABSWIFT as ABSWIFT_torch
from abswift.model.abswift_collator import ABSWIFTCollator


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

def count_parameters(model):
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad:
            continue
        params = parameter.numel()
        total_params += params
    return total_params


class Trainer():
    def __init__(self, cfg:DictConfig, rank_zero_logger: RankZeroLoggingWrapper) -> None:
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()  # call if required elsewhere

        # define model
        model_cfg = OmegaConf.to_container(cfg.model) #convert to regular dict for json serialisation
        self.model = ABSWIFT(
            **model_cfg
        ).to(self.dist.device)

        n_params = count_parameters(self.model)
        rank_zero_logger.log(f'Total number of parameters: {n_params}')

        # distributed data parallel for multi-node training
        if self.dist.world_size > 1:
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )

        #create dataset, sampler, collator, loader
        cfg.dataset.data_dir = to_absolute_path(cfg.dataset.data_dir)
        train_set = InMemoryRandomBuildingsDataset(**cfg.dataset, split = 'train', normalise = True)
        val_set = InMemoryRandomBuildingsDataset(**cfg.dataset, split = 'val', normalise = True)

        collocator = ABSWIFTCollator(
            **cfg.collator,
        )

        self.train_sampler = torch.utils.data.DistributedSampler(
            train_set,
            self.dist.world_size,
            self.dist.rank,
            shuffle=True
        )
        val_sampler = torch.utils.data.DistributedSampler(
            val_set,
            self.dist.world_size,
            self.dist.rank,
            shuffle=False
        )

        self.train_loader = torch.utils.data.DataLoader(
            train_set,
            batch_size = cfg.training.batch_size,
            sampler=self.train_sampler,
            collate_fn=collocator,
            pin_memory = cfg.pin_memory,
            num_workers = cfg.num_workers
        )

        self.val_loader = torch.utils.data.DataLoader(
            val_set,
            batch_size = cfg.training.batch_size,
            sampler=val_sampler,
            collate_fn=collocator,
            pin_memory = cfg.pin_memory,
            num_workers = cfg.num_workers
        )

        #define loss, optimiser, scheduler
        self.criterion = torch.nn.MSELoss()

        self.optimizer = AdamW(
            self.model.parameters(),
            lr=cfg.scheduler.initial_lr,
            weight_decay=cfg.scheduler.weight_decay,
        )

        self.scheduler = lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=cfg.scheduler.initial_lr,
            steps_per_epoch=len(train_set),
            epochs=cfg.training.epochs,
        )

        #load checkpoint
        self.ckpt_args = {
            "path": cfg.training.ckpt_path,
            "optimizer": self.optimizer,
            "scheduler": self.scheduler,
            "models": self.model,
            }
        
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(device=self.dist.device, **self.ckpt_args)

        #list of label keys
        label_keys = [f'volume_{cat}_{field}' for cat in ['anchor', 'query'] for field in cfg.dataset.volume_output_fields]

        # define forward passes for training and inference
        @profile
        @StaticCaptureTraining(
            model=self.model, optim=self.optimizer, logger=rank_zero_logger, use_amp=cfg.use_amp, use_graphs=cfg.use_graphs
        )
        def forward_train(batch):
            #pop labels
            labels = {k:batch.pop(k) for k in label_keys if k in batch.keys()}
            pred = self.model(**batch)
            losses = {
                k:self.criterion(pred[k].squeeze(), labels[k].squeeze()) for k in pred.keys()
            }
            loss = 0
            for v in losses.values():
                loss = loss + v
            return loss

        @profile
        @StaticCaptureEvaluateNoGrad(
            model=self.model, logger=rank_zero_logger, use_amp=cfg.use_amp, use_graphs=False
        )
        def forward_eval(batch):
            labels = {k:batch.pop(k) for k in label_keys if k in batch.keys()}
            pred = self.model(**batch)
            losses = {
                k:self.criterion(pred[k].squeeze(), labels[k].squeeze()) for k in pred.keys()
            }
            loss = 0
            for v in losses.values():
                loss = loss + v
            return loss

        self.forward_train = forward_train
        self.forward_eval = forward_eval

    @profile
    def train_epoch(self, epoch_ind):

        self.train_sampler.set_epoch(epoch_ind)
        
        total_train_loss = 0.0
        for i, batch in enumerate(self.train_loader):
            
            batch = {k:v.to(self.dist.device) for k,v in batch.items()}
            total_train_loss += self.forward_train(batch).detach()
            
            self.scheduler.step()
            
        total_train_loss = total_train_loss / len(self.train_loader)
        
        total_val_loss = 0.0
        for _, batch in enumerate(self.val_loader):

            batch = {k:v.to(self.dist.device) for k,v in batch.items()}
            total_val_loss += self.forward_eval(batch)
        total_val_loss = total_val_loss / len(self.val_loader)

        mem_usage = torch.cuda.memory_reserved(self.dist) / 1024**3

        return total_train_loss, total_val_loss, mem_usage

    def train(self, n_epochs, logger, rank_zero_logger):
        '''Basic training loop'''
        start = time.time()
        rank_zero_logger.info("Training started...")
        for epoch in range(self.epoch_init, n_epochs):
            
            loss, val_loss, mem_usage = self.train_epoch(epoch)
            
            rank_zero_logger.info(
                f"epoch: {epoch}, loss: {loss:10.3e}, val_loss: {val_loss:10.3e}, time per epoch: {(time.time()-start):10.3e}, memory usage: {mem_usage:.2f}Gb"
            )

            # save checkpoint
            if self.dist.world_size > 1:
                torch.distributed.barrier()
            if self.dist.rank == 0:
                #save every 100 epochs
                if epoch % 100 == 99:
                    save_checkpoint(
                        **self.ckpt_args,
                        epoch=epoch,
                    )
                    logger.info(f"Saved model on rank {self.dist.rank}")
            start = time.time()
        rank_zero_logger.info("Training completed!")

@hydra.main(version_base="1.3", config_path="", config_name="config")
def main(cfg: DictConfig) -> None:

    #initialise profiler
    profiler = Profiler()
    if cfg.line_profiler:
        profiler.enable('line_profiler')
    if cfg.torch_profiler:
        profiler.enable('torch')
    profiler.initialize()

    # initialize distributed manager
    DistributedManager.initialize()
    dist = DistributedManager()

    logger = PythonLogger("main")  # General python logger
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)  # Rank 0 logger
    rank_zero_logger.file_logging()

    torch.manual_seed(0)

    trainer = Trainer(cfg, rank_zero_logger)

    trainer.train(cfg.training.epochs, logger, rank_zero_logger)

    profiler.finalize()

if __name__ == "__main__":
    # setup comunication port
    # We define a port based on slurm job id
    # to have different port between different trainings submitted at the same time
    if 'SLURM_JOB_ID' in os.environ:
        os.environ['MASTER_PORT'] = str(1000+int(os.environ['SLURM_JOB_ID'])%1000)
        
    main()