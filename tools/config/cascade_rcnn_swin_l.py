# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
from detectron2.config import LazyCall as L
from detectron2.modeling import SwinTransformer

from .cascade_rcnn_mvitv2_b_in21k import GetModel # pylint: disable=relative-beyond-top-level
from .Config import GetConfigs # pylint: disable=relative-beyond-top-level

IMAGE_SIZE = 1024  # For original model configuration, set this to 1024.
CONV_DIM = 256  # For original model configuration, set this to 256.
BOX_HEADS_FULLY_CONNECTED_DIM = 1024  # For original model configuration, set this to 1024.

dataloader, train = GetConfigs(IMAGE_SIZE)
train.IMAGE_SIZE = IMAGE_SIZE

model = GetModel(IMAGE_SIZE, CONV_DIM, BOX_HEADS_FULLY_CONNECTED_DIM)

model.backbone.bottom_up = L(SwinTransformer)(
    depths=[2, 2, 18, 2],
    drop_path_rate=0.4,
    embed_dim=192,
    num_heads=[6, 12, 24, 48],
)
model.backbone.in_features = ("p0", "p1", "p2", "p3")
