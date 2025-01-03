#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
"""
A main training script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""

import logging
import os
from collections import OrderedDict
from fvcore.common.param_scheduler import MultiStepParamScheduler

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.config import LazyCall as L
from detectron2.data import MetadataCatalog, get_detection_dataset_dicts, build_detection_test_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch
from detectron2.evaluation import (
    CityscapesInstanceEvaluator,
    CityscapesSemSegEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    verify_results,
)
from detectron2.modeling import GeneralizedRCNNWithTTA

from detectron2.data.datasets import register_coco_instances

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.engine.defaults import create_ddp_model
import torch

TEST_BATCH_SIZE = 1

VisionTransformers = {
    "ViT-Base":            "nightowls_rcnn_vitdet_b.py",
    "Cascade-ViT-Base" :   "nightowls_cascade_rcnn_vitdet_b.py",
    "Cascade-MViTv2-Base": "nightowls_cascade_rcnn_mvitv2_b_in21k.py",
}

def build_evaluator(cfg, dataset_name, output_folder=None):
    """
    Create evaluator(s) for a given dataset.
    This uses the special metadata "evaluator_type" associated with each builtin dataset.
    For your own dataset, you can simply create an evaluator manually in your
    script and do not have to worry about the hacky if-else logic here.
    """
    if output_folder is None:
        output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
    evaluator_list = []
    evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
    if evaluator_type in ["sem_seg", "coco_panoptic_seg"]:
        evaluator_list.append(
            SemSegEvaluator(
                dataset_name,
                distributed=True,
                output_dir=output_folder,
            )
        )
    if evaluator_type in ["coco", "coco_panoptic_seg"]:
        evaluator_list.append(COCOEvaluator(dataset_name, output_dir=output_folder))
    if evaluator_type == "coco_panoptic_seg":
        evaluator_list.append(COCOPanopticEvaluator(dataset_name, output_folder))
    if evaluator_type == "cityscapes_instance":
        return CityscapesInstanceEvaluator(dataset_name)
    if evaluator_type == "cityscapes_sem_seg":
        return CityscapesSemSegEvaluator(dataset_name)
    elif evaluator_type == "pascal_voc":
        return PascalVOCDetectionEvaluator(dataset_name)
    elif evaluator_type == "lvis":
        return LVISEvaluator(dataset_name, output_dir=output_folder)
    if len(evaluator_list) == 0:
        raise NotImplementedError(
            "no Evaluator for the dataset {} with the type {}".format(dataset_name, evaluator_type)
        )
    elif len(evaluator_list) == 1:
        return evaluator_list[0]
    return DatasetEvaluators(evaluator_list)

class Trainer(DefaultTrainer):
    """
    We use the "DefaultTrainer" which contains pre-defined default logic for
    standard training workflow. They may not work for you, especially if you
    are working on a new research project. In that case you can write your
    own training loop. You can use "tools/plain_train_net.py" as an example.
    """
    def __init__(self, cfg):
        super().__init__(cfg)

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        return build_evaluator(cfg, dataset_name, output_folder)

    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res

    #
    # Overrides
    #
    @classmethod
    def build_model(cls, cfg):
        if cfg.MODEL.BACKBONE.NAME in VisionTransformers.keys():
            logger = logging.getLogger("detectron2.trainer")
            logger.info(f"Configuring {cfg.MODEL.BACKBONE.NAME} model.")
            model_config = model_zoo.get_config(VisionTransformers[cfg.MODEL.BACKBONE.NAME]).model
            model_config.roi_heads.mask_in_features = None
            model = instantiate(model_config)

            # Freeze some modules.
            # model.backbone.net.requires_grad_(False)        # Freeze vision transformer only for ViT.
            # model.backbone.bottom_up.requires_grad_(False)  # Freeze vision transformer only for MViT.
            model.backbone.requires_grad_(False)              # Freeze full backbone.
            model.proposal_generator.requires_grad_(True)
            model.roi_heads.box_pooler.requires_grad_(True)
        else:
            model = super().build_model(cfg)
        model.to("cuda")
        return model

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        return build_detection_test_loader(cfg, dataset_name, batch_size=TEST_BATCH_SIZE)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.BACKBONE.NAME in VisionTransformers.keys():
            logger = logging.getLogger("detectron2.trainer")
            logger.info(f"Configuring {cfg.MODEL.BACKBONE.NAME} train data loader.")
            train_loader_config = model_zoo.get_config(VisionTransformers[cfg.MODEL.BACKBONE.NAME]).dataloader.train
            train_loader_config.dataset = L(get_detection_dataset_dicts)(names=cfg.DATASETS.TRAIN)
            train_loader_config.total_batch_size = cfg.SOLVER.IMS_PER_BATCH
            return instantiate(train_loader_config)
        else:
            return super().build_train_loader(cfg)

    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.MODEL.BACKBONE.NAME in VisionTransformers.keys():
            logger = logging.getLogger("detectron2.trainer")
            logger.info(f"Configuring {cfg.MODEL.BACKBONE.NAME} optimizer.")
            optimizer_config = model_zoo.get_config(VisionTransformers[cfg.MODEL.BACKBONE.NAME]).optimizer
            optimizer_config.params.model = model
            return instantiate(optimizer_config)
        else:
            return super().build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        if cfg.MODEL.BACKBONE.NAME in VisionTransformers.keys():
            logger = logging.getLogger("detectron2.trainer")
            logger.info(f"Configuring {cfg.MODEL.BACKBONE.NAME} learning rate scheduler.")
            scheduler_config = L(MultiStepParamScheduler)(
                values=[1.0, 0.1, 0.01],
                milestones=cfg.SOLVER.STEPS,
                num_updates=cfg.SOLVER.MAX_ITER,
            )
            return instantiate(scheduler_config)
        else:
            return super().build_lr_scheduler(cfg, optimizer)

def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    register_coco_instances("nightowls_train", {},
                            "nightowls/nightowls_training.json",
                            "nightowls/nightowls_training")
    register_coco_instances("nightowls_val", {},
                            "nightowls/nightowls_validation.json",
                            "nightowls/nightowls_validation")
    register_coco_instances("coco_2017_train_10_plus_shots", {},
                            "datasets/coco/annotations/instances_train2017_10_plus_shots.json",
                            "datasets/coco/train2017")
    # COCO Few Shots
    register_coco_instances("coco_2017_train_50_plus_shots", {},
                            "datasets/coco/annotations/instances_train2017_50_plus_shots.json",
                            "datasets/coco/train2017")
    register_coco_instances("coco_2017_train_100_plus_shots", {},
                            "datasets/coco/annotations/instances_train2017_100_plus_shots.json",
                            "datasets/coco/train2017")
    # NightOwls Few Shots
    number_of_shots = (18, 38, 92, 149, 269, 626)
    for shots in number_of_shots:
        register_coco_instances(f"nightowls_train_{shots}_shots", {},
                                f"nightowls/nightowls_training_{shots}_shots.json",
                                "nightowls/nightowls_training")
    # NightOwls Reannotated Few Shots
    number_of_shots = (193, 336, 799)
    for shots in number_of_shots:
        register_coco_instances(f"nightowls_reannotated_train_{shots}_shots", {},
                                f"nightowls/nightowls_reannotated_train_{shots}_shots.json",
                                "nightowls/nightowls_training")

    cfg = setup(args)

    if args.eval_only:
        model = Trainer.build_model(cfg)
        print(model)
        model = create_ddp_model(model, fp16_compression=True)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=args.resume
        )
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            res = Trainer.test(cfg, model)
            if cfg.TEST.AUG.ENABLED:
                res.update(Trainer.test_with_TTA(cfg, model))
            if comm.is_main_process():
                verify_results(cfg, res)
            return res

    """
    If you'd like to do anything fancier than the standard training logic,
    consider writing your own training loop (see plain_train_net.py) or
    subclassing the trainer.
    """
    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=args.resume)
    if cfg.TEST.AUG.ENABLED:
        trainer.register_hooks(
            [hooks.EvalHook(0, lambda: trainer.test_with_TTA(cfg, trainer.model))]
        )
    return trainer.train()


def invoke_main() -> None:
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )


if __name__ == "__main__":
    invoke_main()  # pragma: no cover
