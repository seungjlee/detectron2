from detectron2.config import LazyCall as L
from detectron2.layers import ShapeSpec
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.roi_heads import (
    FastRCNNOutputLayers,
    FastRCNNConvFCHead,
    CascadeROIHeads,
)

from .nightowls_rcnn_vitdet_b import GetConfigs

IMAGE_SIZE = 320  # For original model configuration, set this to 1024.
TRAIN_BATCH_SIZE = 30
BOX_HEADS_FULLY_CONNECTED_DIM = 1024

# These seem to get overriden so they are set again in the yaml file.
# Need to figure this out.
MAX_ITERATIONS = 60000
SCHEDULER_STEPS = [8000, 52000]

dataloader, lr_multiplier, model, optimizer, train = GetConfigs(
    image_size=IMAGE_SIZE,
    train_batch_size=TRAIN_BATCH_SIZE,
    max_iterations=MAX_ITERATIONS,
    scheduler_steps=SCHEDULER_STEPS,
)

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

# MODEL.WEIGHTS in yaml file used when running train_net.py
train.init_checkpoint = "cascade_mask_rcnn_vitdet_b_model_final_435fa9.pkl"