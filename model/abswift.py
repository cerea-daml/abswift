# Model with a more gradual transition from buildings / ground to volume
# Also, let's make coordinates fully relative

from functools import partial

import torch
from torch import nn

from modules.supernode_pooling_posonly import SupernodePoolingPosonly

from modules.attention import (
    AnchorAttention,
    SharedweightsCrossattnAttention,
    SharedweightsSplitattnAttention,
)
from modules.blocks import TransformerBlock, PerceiverBlock
from modules.continuous_sincos_embed import ContinuousSincosEmbed
from modules.rope_frequency import RopeFrequency
from modules.supernode_pooling_posonly import SupernodePoolingPosonly


class SupernodePoolingRelPos(SupernodePoolingPosonly):
    '''Modification of AB-UPT's supernode pooling layer that takes into account features in addition of coordinates
    Only the relative positions mode is implemented.

    Args:
        radius: Radius around each supernode. From points within this radius, messages are passed to the supernode.
        k: Numer of neighbors for each supernode. From the k-NN points, messages are passed to the supernode.
        hidden_dim: Hidden dimension for positional embeddings, messages and the resulting output vector.
        ndim: Number of positional dimension (e.g., ndim=2 for a 2D position, ndim=3 for a 3D position)
        nfeat: Number of features
        max_degree: Maximum degree of the radius graph. Defaults to 32.'''
    def __init__(self,
                    hidden_dim: int,
                    ndim: int,
                    nfeat: int,
                    radius: float | None = None,
                    k: int | None = None,
                    max_degree: int = 32):
        super().__init__(hidden_dim, ndim, radius, k, max_degree, mode = 'relpos')

        #modify message dim to account for features
        message_input_dim = hidden_dim + nfeat
        
        self.message = nn.Sequential(
            nn.Linear(message_input_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        #also modify projection
        self.proj = nn.Linear( 2 * hidden_dim + nfeat, hidden_dim)

    def create_messages(self, input_pos, src_idx, dst_idx, supernode_idx):
        '''Create messages with features as well as positions'''

        #split positions and features
        input_pos, input_feat = input_pos[... ,:self.ndim], input_pos[..., self.ndim:]
        
        #embed positions
        src_pos = input_pos[src_idx]
        dst_pos = input_pos[dst_idx]
        dist = dst_pos - src_pos
        mag = dist.norm(dim=1).unsqueeze(-1)
        x_pos = self.rel_pos_embed(torch.concat([dist, mag], dim=1))

        #concatenate positions and features
        x = torch.concat([x_pos, input_feat[src_idx]], dim = -1)
        
        supernode_feat = input_feat[supernode_idx]
        supernode_pos_embed = self.pos_embed(input_pos[supernode_idx])
        supernode_embed = torch.concat([supernode_pos_embed, supernode_feat], dim = -1)

        #message
        x = self.message(x)

        return x, supernode_embed

class ABSWIFT(nn.Module):
    '''Anchored Branched Steady-state WInd Flow Transformer'''

    def __init__(
        self,
        #input and output dimensions
        ndim: int = 3, #3D problem
        input_dim_buildings: int = 3, #buildings coordinates, without features
        input_dim_ground: int = 5, #ground coordinates (3D) + 1/Lmo + z0
        input_dim_profile: int = 320, #64 vertical levels * 5 variables (velocity, pressure, k, epsilon, potential_temperature)
        output_dim: dict[str:int] = 6, #velocity (3D), deltaP, k, epsilon
        
        # model
        dim: int = 192,  # dimension of a ViT-tiny
        num_heads: int = 3,  # number of attention heads in a ViT-tiny
        geometry_preprocessor_blocks: str = 'sc', #transformers block encoding ground and buildings branches
        geometry_volume_blocks: str = 'scscsc', #transformers processor of geometry and volume blocks
        n_volume_decoder_blocks: int = 3, #volume only processor

        radius_buildings: float = 1,  # radius for supernode pooling from buildings
        radius_ground: float = 5, #radius for supernode pooling from ground data,
        **kwargs,):
        super().__init__(**kwargs)

        self.ndim = ndim
        self.rope = RopeFrequency(dim=dim // num_heads, ndim=ndim, assert_positive=False)

        #geometry supernode poolings
        self.buildings_supernode_pooling = SupernodePoolingRelPos(
            hidden_dim = dim,
            ndim = ndim,
            nfeat = input_dim_buildings - ndim,
            radius = radius_buildings,
        )
        
        self.ground_supernode_pooling = SupernodePoolingRelPos(
            hidden_dim = dim,
            ndim = ndim,
            nfeat = input_dim_ground - ndim,
            radius = radius_ground,
        )

        #geometry encoder blocks
        self.geometry_blocks = nn.ModuleList()
        for block in geometry_preprocessor_blocks:
            if block == "s":
                # weight-shared self-attention within buildings/ground tokens
                block_ctor = partial(TransformerBlock, attn_ctor=SharedweightsSplitattnAttention)
            elif block == "c":
                # weight-shared cross-attention between buildings/ground tokens
                block_ctor = partial(TransformerBlock, attn_ctor=SharedweightsCrossattnAttention)
            else:
                raise NotImplementedError
            self.geometry_blocks.append(block_ctor(dim=dim, num_heads=num_heads))

        #Profile embedding
        self.profile_encoder = nn.Sequential(
            nn.Linear(input_dim_profile, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        #volume embedding
        self.pos_embed = ContinuousSincosEmbed(dim=dim, ndim=ndim)
        self.volume_bias = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Linear(dim, dim),
        )

        #geometry/volume blocks
        self.geometry_volume_blocks = nn.ModuleList()
        for block in geometry_volume_blocks:
            if block == "s":
                # weight-shared self-attention within geometry/volume tokens
                block_ctor = partial(TransformerBlock, attn_ctor=SharedweightsSplitattnAttention)
            elif block == "c":
                # weight-shared cross-attention between geometry/volume tokens
                block_ctor = partial(TransformerBlock, attn_ctor=SharedweightsCrossattnAttention)
            else:
                raise NotImplementedError
            self.geometry_volume_blocks.append(block_ctor(dim=dim, num_heads=num_heads))

        # volume-specific blocks
        self.volume_blocks = nn.ModuleList(
            [
                TransformerBlock(
                    dim=dim,
                    num_heads=num_heads,
                    attn_ctor=AnchorAttention,
                )
                for _ in range(n_volume_decoder_blocks)
            ],
        )

        #decoder: separate Mlps for each field
        self.decoder = nn.ModuleDict({
            field: nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, out_dim))
            for field, out_dim in output_dim.items()
        })

        # init weights
        # (there are only nn.Linear and nn.LayerNorm in AB-UPT, layernorms are initialized correctly by default)
        def init_weights(m: nn.Module) -> None:
            if isinstance(m, nn.Linear):
                torch.nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

        self.apply(init_weights)


    def forward(self,
                buildings_position: torch.Tensor,
                buildings_supernode_idx: torch.Tensor,
                buildings_batch_idx: torch.Tensor | None,
                ground_data: torch.Tensor,
                ground_supernode_idx: torch.Tensor,
                ground_batch_idx: torch.Tensor | None,
                volume_anchor_position: torch.Tensor,
                profile: torch.Tensor,
                volume_query_position: torch.Tensor | None = None,
                ) -> dict[str, torch.Tensor]:
        
        # create split sizes + optionally concat query positions
        num_anchor_positions = volume_anchor_position.size(1)
        if volume_query_position is None:
            pos_volume_all = volume_anchor_position
            #split between buildings + ground, volume anchors. No query for either geom or volume
            geometry_volume_split = [buildings_supernode_idx.size(0) + ground_supernode_idx.size(0), 
                                    volume_anchor_position.size(1)]
        else:
            pos_volume_all = torch.concat([volume_anchor_position, volume_query_position], dim = 1)
            #split between buildings + ground, none, volume anchors, and volume queries. 'None' as in there is no query points for geometry
            geometry_volume_split = [buildings_supernode_idx.size(0) + ground_supernode_idx.size(0), 
                                    0,
                                    volume_anchor_position.size(1),
                                    volume_query_position.size(1)]

        #Split between building and ground
        geometry_split = [buildings_supernode_idx.size(0), ground_supernode_idx.size(0)] #Split between buildings and ground

        # rope frequencies
        assert buildings_batch_idx is None or buildings_batch_idx.unique().numel() == 1, "batch_size > 1 not supported"
        assert ground_batch_idx is None or ground_batch_idx.unique().numel() == 1, "batch_size > 1 not supported"
        building_rope = self.rope(buildings_position[buildings_supernode_idx, :self.ndim].unsqueeze(0))
        ground_rope = self.rope(ground_data[ground_supernode_idx, :self.ndim].unsqueeze(0))
        geometry_rope_all = torch.concat([building_rope, ground_rope], dim=1)
        volume_rope_all = self.rope(pos_volume_all)
        rope_all = torch.concat([geometry_rope_all, volume_rope_all], dim = 1)

        #encode geometries
        x_buildings = self.buildings_supernode_pooling(
            input_pos = buildings_position,
            supernode_idx = buildings_supernode_idx,
            batch_idx = buildings_batch_idx)
        
        x_ground = self.ground_supernode_pooling(
            input_pos = ground_data,
            supernode_idx = ground_supernode_idx,
            batch_idx = ground_batch_idx)
        
        #concatenate sequences for high GPU usage
        x_geom = torch.concat([x_buildings, x_ground], dim = 1)
        #apply ground/buildings blocks
        for block in self.geometry_blocks:
            x_geom = block(x_geom, attn_kwargs = dict(split_size = geometry_split, freqs = geometry_rope_all))

        #encode profile
        x_profile = self.profile_encoder(profile.flatten())

        #encode volume
        x_volume = self.pos_embed(pos_volume_all)
        x_volume = self.volume_bias(x_volume)

        #apply profile embedings to volume points
        x_profile = torch.expand_copy(x_profile, (pos_volume_all.shape[1], -1)).unsqueeze(0)
        x_volume = x_volume + x_profile

        #geometry-volume blocks. Sequences are concatenated for high GPU usage
        x = torch.concat([x_geom, x_volume], dim = 1)
        for block in self.geometry_volume_blocks:
            x = block(x, attn_kwargs = dict(split_size = geometry_volume_split, freqs = rope_all))

        #split between geometry and volume points. We discard geometry embeddings at this point
        if volume_query_position is None:
            _, x_volume = x.split([geometry_volume_split[0], geometry_volume_split[1]], dim = 1)
        else:
            _, x_volume = x.split([geometry_volume_split[0], geometry_volume_split[2] + geometry_volume_split[3]], dim = 1)

        #apply volume-specific blocks
        for block in self.volume_blocks:
            x_volume = block(x_volume, attn_kwargs = dict(num_anchor_tokens = volume_anchor_position.size(1), freqs = volume_rope_all))

        #apply decoder
        output = {field:decoder(x_volume) for field, decoder in self.decoder.items()}

        #split between query and anchor for final output
        if volume_query_position is None:
            outputs = {f'volume_anchor_{field}':v for field,v in output.items()}
        else:
            outputs_anchor = {f'volume_anchor_{field}':v[:, :num_anchor_positions] for field,v in output.items()}
            outputs_query = {f'volume_query_{field}':v[:, num_anchor_positions:] for field,v in output.items()}
            outputs = {**outputs_anchor, **outputs_query}

        return outputs