# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
from functools import partial
import torch.nn as nn

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.modeling import MViT
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    #CascadeROIHeads,
)

from .Config import GetConfigs # pylint: disable=relative-beyond-top-level
from .RoiHeads import CascadeROIHeadsPlus # pylint: disable=relative-beyond-top-level

IMAGE_SIZE = 1024  # For original model configuration, set this to 1024.
CONV_DIM = 256  # For original model configuration, set this to 256.
BOX_HEADS_FULLY_CONNECTED_DIM = 1024  # For original model configuration, set this to 1024.

dataloader, train = GetConfigs(IMAGE_SIZE)
train.IMAGE_SIZE = IMAGE_SIZE

def GetModel(image_size, conv_dim, box_heads_fully_connected_dim, cascade_roi_heads = True):
    Model = model_zoo.get_config("common/models/mask_rcnn_fpn.py").model
    constants = model_zoo.get_config("common/data/constants.py").constants

    Model.pixel_mean = constants.imagenet_rgb256_mean
    Model.pixel_std = constants.imagenet_rgb256_std
    Model.input_format = "RGB"
    Model.backbone.bottom_up = L(MViT)(
        embed_dim=96,
        depth=24,
        num_heads=1,
        last_block_indexes=(1, 4, 20, 23),
        residual_pooling=True,
        drop_path_rate=0.4,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        out_features=("scale2", "scale3", "scale4", "scale5"),
    )
    Model.backbone.in_features = "${.bottom_up.out_features}"
    Model.backbone.square_pad = image_size

    # New heads and LN
    Model.backbone.norm = "LN"  # Use LN in FPN
    Model.roi_heads.box_head.conv_norm = Model.roi_heads.mask_head.conv_norm = "LN"

    # 2conv in RPN:
    Model.proposal_generator.head.conv_dims = [-1, -1]

    # arguments that don't exist for Cascade R-CNN
    if cascade_roi_heads:
        [Model.roi_heads.pop(k) for k in ["box_head", "box_predictor", "proposal_matcher"]] # pylint: disable=expression-not-assigned

        Model.roi_heads.update(
            _target_=CascadeROIHeadsPlus,
            box_heads=[
                L(FastRCNNConvFCHead)(
                    input_shape=ShapeSpec(channels=256, height=7, width=7),
                    conv_dims=[conv_dim, conv_dim, conv_dim, conv_dim],
                    fc_dims=[box_heads_fully_connected_dim],
                    conv_norm="LN",
                )
                for _ in range(3)
            ],
            box_predictors=[
                L(FastRCNNOutputLayers)(
                    input_shape=ShapeSpec(channels=box_heads_fully_connected_dim),
                    test_score_thresh=0.45,
                    box2box_transform=L(Box2BoxTransform)(weights=(w1, w1, w2, w2)),
                    cls_agnostic_bbox_reg=True,
                    num_classes="${...num_classes}",
                )
                for (w1, w2) in [(10, 5), (20, 10), (30, 15)]
            ],
            proposal_matchers=[
                L(Matcher)(thresholds=[th], labels=[0, 1], allow_low_quality_matches=False)
                # for th in [0.6, 0.75, 0.9]
                for th in [0.5, 0.6, 0.7]
            ],
        )

    return Model

model = GetModel(IMAGE_SIZE, CONV_DIM, BOX_HEADS_FULLY_CONNECTED_DIM)
