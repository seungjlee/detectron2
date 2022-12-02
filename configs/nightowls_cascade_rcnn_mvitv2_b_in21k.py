from functools import partial
import torch.nn as nn
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling import MViT
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    CascadeROIHeads,
)

# from ..common.coco_loader_lsj import dataloader
import detectron2.data.transforms as T

TRAIN_BATCH_SIZE = 20  # 20*7500 = 150000 ~ 1 epoch
IMAGE_SIZE = 512  # For original model configuration, set this to 1024.
BOX_HEADS_FULLY_CONNECTED_DIM = 1024  # For original model configuration, set this to 1024.

# These seem to get overriden so they are set again in nightowls_mvitv2.yaml.
# Need to figure this out.
MAX_ITERATIONS = 100000  # 100000 ~ 10 epochs
SCHEDULER_STEPS = [20000, 80000]

#region Data Loader
# Data using LSJ
dataloader = model_zoo.get_config("nightowls.py").dataloader
dataloader.train.mapper.augmentations = [
    L(T.RandomFlip)(horizontal=True),  # flip first
    L(T.ResizeScale)(
        min_scale=0.1, max_scale=2.0, target_height=IMAGE_SIZE, target_width=IMAGE_SIZE
    ),
    L(T.FixedSizeCrop)(crop_size=(IMAGE_SIZE, IMAGE_SIZE), pad=False),
]
dataloader.train.mapper.image_format = "RGB"
dataloader.train.mapper.use_instance_mask = False
dataloader.train.total_batch_size = TRAIN_BATCH_SIZE
dataloader.train.num_workers = 1
dataloader.train.mapper.recompute_boxes = False

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE),
]
#endregion

model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
constants = model_zoo.get_config("common/data/constants.py").constants
model.pixel_mean = constants.imagenet_rgb256_mean
model.pixel_std = constants.imagenet_rgb256_std
model.input_format = "RGB"
model.backbone.bottom_up = L(MViT)(
    embed_dim=96,
    depth=24,
    num_heads=1,
    last_block_indexes=(1, 4, 20, 23),
    residual_pooling=True,
    drop_path_rate=0.4,
    norm_layer=partial(nn.LayerNorm, eps=1e-6),
    out_features=("scale2", "scale3", "scale4", "scale5"),
)
model.backbone.in_features = "${.bottom_up.out_features}"
model.backbone.square_pad = IMAGE_SIZE

# New heads and LN
model.backbone.norm = "LN"  # Use LN in FPN
model.roi_heads.box_head.conv_norm = model.roi_heads.mask_head.conv_norm = "LN"

# 2conv in RPN:
model.proposal_generator.head.conv_dims = [-1, -1]

# arguments that don't exist for Cascade R-CNN
[model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]]
model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[256, 256, 256, 256],
            fc_dims=[BOX_HEADS_FULLY_CONNECTED_DIM],
            conv_norm="LN",
        )
        for _ in range(3)
    ],
    box_predictors=[
        L(FastRCNNOutputLayers)(
            input_shape=ShapeSpec(channels=BOX_HEADS_FULLY_CONNECTED_DIM),
            test_score_thresh=0.05,
            box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
            cls_agnostic_bbox_reg=True,
            num_classes="${...num_classes}",
        )
        for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
    ],
    proposal_matchers=[
        L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
        for th in [0.5, 0.6, 0.7]
    ],
)

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = ""

# Schedule
train.max_iter = MAX_ITERATIONS

lr_multiplier = L(WarmupParamScheduler)(
    scheduler=L(MultiStepParamScheduler)(
        values=[1.0, 0.1, 0.01],
        milestones=SCHEDULER_STEPS,
        num_updates=train.max_iter,
    ),
    warmup_length=250 / train.max_iter,
    warmup_factor=0.001,
)

optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
optimizer.lr = 8e-5
