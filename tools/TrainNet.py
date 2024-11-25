#!/usr/bin/python3
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
# pylint: disable=line-too-long
# %%
import warnings

import numpy as np
import torch

import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import default_argument_parser, launch
from detectron2.engine.defaults import create_ddp_model
from detectron2.evaluation import verify_results

from Model.trainer import Trainer

# %%
def main(arguments):
    np.set_printoptions(floatmode="fixed", precision=5, linewidth=160)
    try:
        register_coco_instances("coco_2017_train_100", {},
                                "datasets/instances_train2017_100_plus_shots.json",
                                "datasets/coco/train2017")
        register_coco_instances("coco_2017_train_500", {},
                                "datasets/instances_train2017_500_plus_shots.json",
                                "datasets/coco/train2017")
        register_coco_instances("coco_2017_val_plus", {},
                                "datasets/instances_val2017_plus.json",
                                "datasets/coco/val2017")
    except AssertionError:
        print("Data already registered.")

    cfg = Trainer.setup(arguments)

    if arguments.eval_only:
        model = Trainer.build_model(cfg)
        # print(model)
        model = create_ddp_model(model, fp16_compression=True)
        DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
            cfg.MODEL.WEIGHTS, resume=arguments.resume
        )
        model.eval()
        with torch.no_grad(), torch.amp.autocast("cuda"):
            res = Trainer.test(cfg, model)
            if comm.is_main_process():
                verify_results(cfg, res)
            return res

    trainer = Trainer(cfg)
    trainer.resume_or_load(resume=arguments.resume)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        return trainer.train()

if __name__ == "__main__":
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
