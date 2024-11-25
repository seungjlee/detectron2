# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
from .cascade_rcnn_mvitv2_b_in21k import GetModel # pylint: disable=relative-beyond-top-level
from .Config import GetConfigs # pylint: disable=relative-beyond-top-level

IMAGE_SIZE = 800  # For original model configuration, set this to 1024.
CONV_DIM = 256  # For original model configuration, set this to 256.
BOX_HEADS_FULLY_CONNECTED_DIM = 1024  # For original model configuration, set this to 1024.

dataloader, train = GetConfigs(IMAGE_SIZE)
train.IMAGE_SIZE = IMAGE_SIZE

model = GetModel(IMAGE_SIZE, CONV_DIM, BOX_HEADS_FULLY_CONNECTED_DIM)

model.backbone.bottom_up.embed_dim = 192
model.backbone.bottom_up.depth = 80
model.backbone.bottom_up.num_heads = 3
model.backbone.bottom_up.last_block_indexes = (3, 11, 71, 79)
model.backbone.bottom_up.drop_path_rate = 0.6
model.backbone.bottom_up.use_act_checkpoint = True
