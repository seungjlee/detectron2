# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name

from .Config import GetConfigs # pylint: disable=relative-beyond-top-level
from .MViT_V2_Base import GetModel # pylint: disable=relative-beyond-top-level

IMAGE_SIZE = 1280  # For original model configuration, set this to 1024.
CONV_DIM = 256  # For original model configuration, set this to 256.
BOX_HEADS_FULLY_CONNECTED_DIM = 1024  # For original model configuration, set this to 1024.

dataloader, train = GetConfigs(IMAGE_SIZE)
train.IMAGE_SIZE = IMAGE_SIZE

model = GetModel(IMAGE_SIZE, CONV_DIM, BOX_HEADS_FULLY_CONNECTED_DIM, cascade_roi_heads=False)
