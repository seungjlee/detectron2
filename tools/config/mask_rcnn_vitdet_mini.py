# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    MaskRCNNConvUpsampleHead,
)
from .Config import GetConfigs # pylint: disable=relative-beyond-top-level

IMAGE_SIZE = 1024  # For original model configuration, set this to 1024.

BOX_HEAD_CONVOLUTION_DIM = 128  # For original model configuration, set this to 256.
BOX_HEAD_CONVOLUTIONS = 2  # For original model configuration, set this to 4.
BOX_HEAD_FULLY_CONNECTED_DIM = 512  # For original model configuration, set this to 1024.

MAX_HEAD_CONVOLUTION_DIM = 128  # For original model configuration, set this to 256.
MAX_HEAD_CONVOLUTIONS = 2  # For original model configuration, set this to 4.

dataloader, train = GetConfigs(image_size=IMAGE_SIZE)
train.IMAGE_SIZE = IMAGE_SIZE

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.backbone.net.img_size = IMAGE_SIZE
model.backbone.square_pad = IMAGE_SIZE

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
        conv_dims=[MAX_HEAD_CONVOLUTION_DIM] * (MAX_HEAD_CONVOLUTIONS + 1),
        conv_norm="LN",
    ),
)
#model.box_predictor.input_shape.channels = BOX_HEADS_FULLY_CONNECTED_DIM
# model.roi_heads=L(StandardROIHeads)(
#     num_classes=80,
#     batch_size_per_image=512,
#     positive_fraction=0.25,
#     proposal_matcher=L(Matcher)(
#         thresholds=[0.5], labels=[0, 1], allow_low_quality_matches=False
#     ),
#     box_in_features=["p2", "p3", "p4", "p5"],
#     box_pooler=L(ROIPooler)(
#         output_size=7,
#         scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
#         sampling_ratio=0,
#         pooler_type="ROIAlignV2",
#     ),
#     box_head=L(FastRCNNConvFCHead)(
#         input_shape=ShapeSpec(channels=256, height=7, width=7),
#         conv_dims=[],
#         fc_dims=[1024, 1024],
#     ),
#     box_predictor=L(FastRCNNOutputLayers)(
#         input_shape=ShapeSpec(channels=1024),
#         test_score_thresh=0.05,
#         box2box_transform=L(Box2BoxTransform)(weights=(10, 10, 5, 5)),
#         num_classes="${..num_classes}",
#     ),
#     mask_in_features=["p2", "p3", "p4", "p5"],
#     mask_pooler=L(ROIPooler)(
#         output_size=14,
#         scales=(1.0 / 4, 1.0 / 8, 1.0 / 16, 1.0 / 32),
#         sampling_ratio=0,
#         pooler_type="ROIAlignV2",
#     ),
#     mask_head=L(MaskRCNNConvUpsampleHead)(
#         input_shape=ShapeSpec(channels=256, width=14, height=14),
#         num_classes="${..num_classes}",
#         conv_dims=[256, 256, 256, 256, 256],
#     ),
# ),