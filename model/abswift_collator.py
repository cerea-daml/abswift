#Collator


from typing import Any

import torch

from collators import ConcatSparseTensorCollator, FieldDecoderCollator, MultiStageCollator
from preprocessors.preprocessor import Preprocessor
from preprocessors import (
    PointSamplingPreprocessor,
    SupernodeSamplingPreprocessor,
    AnchorPointSamplingPreprocessor,
)
    
class ConcatFieldsPreprocessor(Preprocessor):
    def __init__(self, items: list[str], target_key:str, concat_dim:int):
        '''
        Concatenates several fields (eg positions and features) together into a single field.
            Args:
                items:  Which field items should be collated.
                target_key: name of the new field
                concat_dim: dimension at which to concatenate
        '''
        self.items = items
        self.target_key = target_key
        self.concat_dim = concat_dim
    def __call__(self, samples: list[dict[str, Any]]) -> list[dict[str, Any]]:
        new_samples = []
        for sample in samples:
            new_sample = dict(sample)
            fields = [new_sample.pop(key) for key in self.items]
            field = torch.concat(fields, dim = self.concat_dim)
            new_sample[self.target_key] = field
            new_samples.append(new_sample)
        return new_samples

class ABSWIFTCollator(MultiStageCollator):
    '''Collator for atmospheric simulations, for use with AB-SWIFT model
    Applies point subsampling, supernodes selection and anchor selection'''
    def __init__(
        self,
        # Number of nupernodes, anchors, ...
        num_buildings_points: int,
        num_ground_points: int,
        num_buildings_supernodes: int,
        num_ground_supernodes: int,
        num_volume_anchor_points: int,
        volume_fields: list[str],
        #others
        use_query_positions: bool = False,
        seed: int | None = None,
    ):

        super().__init__(
            preprocessors=[
                # All normalisation is done directly in the dataset for better reproducibility across models
                # preprocess geometry data
                PointSamplingPreprocessor(
                    items={"buildings_position"},
                    num_points=num_buildings_points,
                    seed=None if seed is None else seed + 1,
                ),
                PointSamplingPreprocessor(
                    items={"ground_position", "ground_feature"},
                    num_points=num_ground_points,
                    seed=None if seed is None else seed + 2,
                ),
                SupernodeSamplingPreprocessor(
                    item="buildings_position",
                    num_supernodes=num_buildings_supernodes,
                    supernode_idx_key="buildings_supernode_idx",
                    seed=None if seed is None else seed + 3,
                ),
                SupernodeSamplingPreprocessor(
                    item="ground_position",
                    num_supernodes=num_ground_supernodes,
                    supernode_idx_key="ground_supernode_idx",
                    seed=None if seed is None else seed + 4,
                ),
                # subsample volume data
                AnchorPointSamplingPreprocessor(
                    items={"volume_position", *[f"volume_{field}" for field in volume_fields]}, #FIXME: _ is already used to split prefix and postfix, here we run into an issue because we also use it for log_field
                    num_points=num_volume_anchor_points,
                    keep_queries=use_query_positions,
                    to_prefix_and_postfix=lambda item: item.split("_"),
                    to_prefix_midfix_postfix=lambda item: item.split("_") if len(item.split("_")) == 3 else [None] * 3,
                    seed=None if seed is None else seed + 5,
                ),
                #concatenate ground features and positions
                ConcatFieldsPreprocessor(
                    items = ['ground_position', 'ground_feature'],
                    target_key = 'ground_data',
                    concat_dim=-1
                ),
            ],
            collators=[
                # collate geometry positions (remains sparse for supernode_pooling)
                ConcatSparseTensorCollator(
                    items=["buildings_position"],
                    create_batch_idx=True,
                    batch_idx_key="buildings_batch_idx",
                ),
                ConcatSparseTensorCollator(items=["buildings_supernode_idx"], create_batch_idx=False),
                ConcatSparseTensorCollator(
                    items=["ground_data"],
                    create_batch_idx=True,
                    batch_idx_key="ground_batch_idx",
                ),
                ConcatSparseTensorCollator(items=["ground_supernode_idx"], create_batch_idx=False),
                # collate volume data
                FieldDecoderCollator(
                    position_item="volume_anchor_position",
                    target_items=[f"volume_anchor_{field}" for field in volume_fields],
                ),
                # collate auxiliary data
                FieldDecoderCollator(
                    position_item="volume_query_position",
                    target_items=[f"volume_query_{field}" for field in volume_fields],
                    optional=True,
                ),
                # collate profile data
                ConcatSparseTensorCollator(
                    items=["profile"], create_batch_idx=False),
            ],
            postprocessors=[],
            dataset=None,
        )

    def preprocess_inputs_only(self, samples):
        batch = self(samples)
        for key in list(batch.keys()):
            if "field" in key:
                batch.pop(key)
        return batch