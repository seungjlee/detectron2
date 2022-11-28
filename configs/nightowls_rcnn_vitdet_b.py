from functools import partial
from fvcore.common.param_scheduler import MultiStepParamScheduler

from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.solver import WarmupParamScheduler
from detectron2.modeling.backbone.vit import get_vit_lr_decay_rate

#from ..common.coco_loader_lsj import dataloader
import detectron2.data.transforms as T

IMAGE_SIZE = 640  # For original model configuration, set this to 1024.
TRAIN_BATCH_SIZE = 24  # 24*6200 = 148800 ~ 1 epoch

# These seem to get overriden so they are set again in nightowls.yaml.
# Need to figure this out.
MAX_ITERATIONS = 62000  # 62000 ~ 10 epochs
SCHEDULER_STEPS = [12400, 49600]

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
dataloader.train.total_batch_size = TRAIN_BATCH_SIZE
dataloader.train.num_workers = 1
dataloader.train.mapper.recompute_boxes = False

dataloader.test.mapper.augmentations = [
    L(T.ResizeShortestEdge)(short_edge_length=IMAGE_SIZE, max_size=IMAGE_SIZE),
]
#endregion

model = model_zoo.get_config("common/models/mask_rcnn_vitdet.py").model
model.backbone.net.img_size = IMAGE_SIZE
model.backbone.square_pad = IMAGE_SIZE

# Initialization and trainer settings
train = model_zoo.get_config("common/train.py").train
train.amp.enabled = True
train.ddp.fp16_compression = True
train.init_checkpoint = "mask_rcnn_vitdet_b_model_final_61ccd1.pkl"

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

# Optimizer
optimizer = model_zoo.get_config("common/optim.py").AdamW
optimizer.params.lr_factor_func = partial(get_vit_lr_decay_rate, num_layers=12, lr_decay_rate=0.7)
optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
