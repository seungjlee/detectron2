# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
from .Config import GetConfigs # pylint: disable=relative-beyond-top-level
from .MViT_V2_Base import GetModel # pylint: disable=relative-beyond-top-level

import torch.nn as nn
from functools import partial
from detectron2.config import LazyCall as L
from detectron2.layers.batch_norm import NaiveSyncBatchNorm
from detectron2.modeling import MViT

IMAGE_SIZE = 1024  # For original model configuration, set this to 1024.
CONV_DIM = 256  # For original model configuration, set this to 256.
BOX_HEADS_FULLY_CONNECTED_DIM = 1024  # For original model configuration, set this to 1024.

dataloader, train = GetConfigs(IMAGE_SIZE)
train.IMAGE_SIZE = IMAGE_SIZE

model = GetModel(IMAGE_SIZE, CONV_DIM, BOX_HEADS_FULLY_CONNECTED_DIM, cascade_roi_heads=False)

model.backbone.bottom_up.depth = 10
model.backbone.bottom_up.last_block_indexes = (0, 2, 7, 9)
model.backbone.bottom_up.drop_path_rate = 0.2

model.backbone.norm = ""
model.roi_heads.mask_head.conv_norm = None
model.proposal_generator.head.conv_dims = [-1]
