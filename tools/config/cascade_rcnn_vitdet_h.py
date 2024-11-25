# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    CascadeROIHeads,
)
from .Config import GetConfigs # pylint: disable=relative-beyond-top-level
from .cascade_rcnn_vitdet_b import model # pylint: disable=relative-beyond-top-level

IMAGE_SIZE = 1024  # For original model configuration, set this to 1024.
CONV_DIM = 256  # For original model configuration, set this to 256.
BOX_HEADS_FULLY_CONNECTED_DIM = 1024  # For original model configuration, set this to 1024.

dataloader, train = GetConfigs(image_size=IMAGE_SIZE)
train.IMAGE_SIZE = IMAGE_SIZE

model.backbone.net.img_size = IMAGE_SIZE
model.backbone.square_pad = IMAGE_SIZE

model.backbone.net.embed_dim = 1280
model.backbone.net.depth = 32
model.backbone.net.num_heads = 16
model.backbone.net.drop_path_rate = 0.5
# 7, 15, 23, 31 for global attention
model.backbone.net.window_block_indexes = (
    list(range(0, 7)) + list(range(8, 15)) + list(range(16, 23)) + list(range(24, 31))
)

model.roi_heads.update(
    _target_=CascadeROIHeads,
    box_heads=[
        L(FastRCNNConvFCHead)(
            input_shape=ShapeSpec(channels=256, height=7, width=7),
            conv_dims=[CONV_DIM, CONV_DIM, CONV_DIM, CONV_DIM],
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
