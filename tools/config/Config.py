# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
from omegaconf import OmegaConf

import detectron2.data.transforms as T
from detectron2 import model_zoo
from detectron2.config import LazyCall as L
from detectron2.data import (
    DatasetMapper,
    build_detection_test_loader,
    build_detection_train_loader,
    get_detection_dataset_dicts,
)
from detectron2.evaluation import COCOEvaluator

def GetConfigs(image_size):
    dataloader = OmegaConf.create()

    dataloader.train = L(build_detection_train_loader)(
        dataset=L(get_detection_dataset_dicts)(names="coco_2017_train"),
        mapper=L(DatasetMapper)(
            is_train=True,
            augmentations=[
                L(T.RandomFlip)(horizontal=True, vertical=False),
                #L(T.RandomRotation)(angle=(0,45,90,135), sample_style="choice"),
                L(T.FixedSizeCrop)(crop_size=(image_size, image_size), pad=True),
                #L(T.RandomContrast)(intensity_min=0.8, intensity_max=1.2),
                #L(T.RandomBrightness)(intensity_min=0.8, intensity_max=1.2),
            ],
            image_format="RGB",
            use_instance_mask=True,
            instance_mask_format="polygon",
            recompute_boxes = True,
        ),
        total_batch_size=4,
        num_workers=2,
    )

    dataloader.test = L(build_detection_test_loader)(
        dataset=L(get_detection_dataset_dicts)(names="coco_2017_val", filter_empty=False),
        mapper=L(DatasetMapper)(
            is_train=False,
            augmentations=[
                L(T.ResizeShortestEdge)(short_edge_length=image_size, max_size=image_size),
            ],
            image_format="${...train.mapper.image_format}",
            use_instance_mask=True,
            instance_mask_format="polygon",
            recompute_boxes = True,
        ),
        num_workers=2,
    )

    dataloader.evaluator = L(COCOEvaluator)(
        dataset_name="${..test.dataset.names}",
    )
    dataloader.evaluator.tasks = ["bbox"]

    # Initialization and trainer settings
    train = model_zoo.get_config("common/train.py").train
    train.amp.enabled = True
    train.ddp.fp16_compression = True
    train.init_checkpoint = ""
    train.max_iter = 1

    return dataloader, train
