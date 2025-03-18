# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
from .Config import GetConfigs # pylint: disable=relative-beyond-top-level
from .MViT_V2_Base import GetModel # pylint: disable=relative-beyond-top-level

from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    MaskRCNNConvUpsampleHead,
)

IMAGE_SIZE = 1024  # For original model configuration, set this to 1024.

BOX_HEAD_CONVOLUTION_DIM = 128  # For original model configuration, set this to 256.
BOX_HEAD_CONVOLUTIONS = 2  # For original model configuration, set this to 4.
BOX_HEAD_FULLY_CONNECTED_DIM = 192  # For original model configuration, set this to 1024.

MASK_HEAD_CONVOLUTION_DIM = 192  # For original model configuration, set this to 256.
MASK_HEAD_CONVOLUTIONS = 4  # For original model configuration, set this to 4.

dataloader, train = GetConfigs(IMAGE_SIZE)
train.IMAGE_SIZE = IMAGE_SIZE

model = GetModel(IMAGE_SIZE, BOX_HEAD_CONVOLUTION_DIM, BOX_HEAD_FULLY_CONNECTED_DIM, cascade_roi_heads=False)

model.backbone.bottom_up.depth = 10
model.backbone.bottom_up.last_block_indexes = (0, 2, 7, 9)
model.backbone.bottom_up.out_features=("scale2", "scale3", "scale4", "scale5")
model.backbone.bottom_up.drop_path_rate = 0.2

model.backbone.norm = ""
model.roi_heads.mask_head.conv_norm = None
model.proposal_generator.head.conv_dims = [-1]

model.roi_heads.update(
    box_head=L(FastRCNNConvFCHead)(
        input_shape=ShapeSpec(channels=256, height=7, width=7),
        conv_dims=[BOX_HEAD_CONVOLUTION_DIM] * BOX_HEAD_CONVOLUTIONS,
        fc_dims=[BOX_HEAD_FULLY_CONNECTED_DIM],
        conv_norm="LN",
    ),
    box_predictor=L(FastRCNNOutputLayers)(
        input_shape=ShapeSpec(channels=BOX_HEAD_FULLY_CONNECTED_DIM),
        test_score_thresh=0.05,
        box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
        num_classes="${..num_classes}",
    ),
    mask_head=L(MaskRCNNConvUpsampleHead)(
        input_shape=ShapeSpec(channels=256, width=14, height=14),
        num_classes="${..num_classes}",
        conv_dims=[MASK_HEAD_CONVOLUTION_DIM] * (MASK_HEAD_CONVOLUTIONS + 1),
        conv_norm="LN",
    ),
)