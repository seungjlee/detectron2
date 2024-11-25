# pylint: disable=missing-class-docstring
# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=invalid-name
# pylint: disable=line-too-long
import logging
import os
from datetime import datetime

from fvcore.common.param_scheduler import MultiStepParamScheduler
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import LazyCall as L
from detectron2.config import LazyConfig, get_cfg
from detectron2.data import get_detection_dataset_dicts, build_detection_test_loader
from detectron2.engine import BestCheckpointer, DefaultTrainer, default_setup

from detectron2.evaluation import (
    COCOEvaluator,
    DatasetEvaluators,
)

from detectron2.modeling.anchor_generator import DefaultAnchorGenerator
from detectron2.modeling.box_regression import Box2BoxTransform
from detectron2.modeling.matcher import Matcher
from detectron2.modeling.proposal_generator import RPN, StandardRPNHead

from detectron2 import model_zoo
from detectron2.config import instantiate
from detectron2.utils.analysis import parameter_count_table

LAZY_CONFIG_PATH = "./config/"

LazyConfigurations = {
    "cascade_rcnn_mvitv2_b_in21k": f"{LAZY_CONFIG_PATH}cascade_rcnn_mvitv2_b_in21k.py",
    "cascade_rcnn_mvitv2_l_in21k": f"{LAZY_CONFIG_PATH}cascade_rcnn_mvitv2_l_in21k.py",
    "cascade_rcnn_mvitv2_h_in21k": f"{LAZY_CONFIG_PATH}cascade_rcnn_mvitv2_h_in21k.py",
    "cascade_rcnn_swin_l": f"{LAZY_CONFIG_PATH}cascade_rcnn_swin_l.py",
    "cascade_rcnn_vitdet_b": f"{LAZY_CONFIG_PATH}cascade_rcnn_vitdet_b.py",
    "cascade_rcnn_vitdet_l": f"{LAZY_CONFIG_PATH}cascade_rcnn_vitdet_l.py",
    "cascade_rcnn_vitdet_h": f"{LAZY_CONFIG_PATH}cascade_rcnn_vitdet_h.py",
}

class Trainer(DefaultTrainer):

    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        coco_evaluator = COCOEvaluator(dataset_name, tasks=None, output_dir=output_folder)
        return DatasetEvaluators([coco_evaluator])

    @classmethod
    def build_model(cls, cfg):
        logger = logging.getLogger("detectron2.trainer")
        if cfg.MODEL.BACKBONE.NAME in LazyConfigurations:
            logger.info("Configuring %s model.", cfg.MODEL.BACKBONE.NAME)
            model_config = Trainer.GetLazyConfig(cfg).model
            model_config.roi_heads.batch_size_per_image = cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE
            model_config.roi_heads.num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES

            head_conv_dims = model_config.proposal_generator.head.conv_dims
            model_config.proposal_generator.update(L(RPN)(
                in_features=cfg.MODEL.RPN.IN_FEATURES,
                head=L(StandardRPNHead)(in_channels=256, num_anchors=3, conv_dims=head_conv_dims),
                anchor_generator=L(DefaultAnchorGenerator)(
                    sizes=cfg.MODEL.ANCHOR_GENERATOR.SIZES,
                    aspect_ratios=cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS,
                    strides=[4, 8, 16, 32, 64],
                    offset=0.0,
                ),
                anchor_matcher=L(Matcher)(
                    thresholds=cfg.MODEL.RPN.IOU_THRESHOLDS,
                    labels=cfg.MODEL.RPN.IOU_LABELS,
                    allow_low_quality_matches=True
                ),
                box2box_transform=L(Box2BoxTransform)(weights=[1.0, 1.0, 1.0, 1.0]),
                batch_size_per_image=cfg.MODEL.RPN.BATCH_SIZE_PER_IMAGE, # Number of regions per image used to train RPN
                positive_fraction=cfg.MODEL.RPN.POSITIVE_FRACTION, # Target fraction of foreground (positive) examples per RPN minibatch
                pre_nms_topk=(cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN, cfg.MODEL.RPN.PRE_NMS_TOPK_TEST),
                post_nms_topk=(cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN, cfg.MODEL.RPN.POST_NMS_TOPK_TEST),
                nms_thresh=cfg.MODEL.RPN.NMS_THRESH,
            ))

            # logger.info(model_config.roi_heads)
            # model_config.roi_heads.mask_in_features = None
            model = instantiate(model_config)
            if "cascade" in cfg.MODEL.BACKBONE.NAME:
                for box_predictor in model.roi_heads.box_predictor:
                    box_predictor.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
                    box_predictor.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
                    box_predictor.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
            else:
                model.roi_heads.box_predictor.test_score_thresh = cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST
                model.roi_heads.box_predictor.test_topk_per_image = cfg.TEST.DETECTIONS_PER_IMAGE
                model.roi_heads.box_predictor.test_nms_thresh = cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST
                logger.info((model.roi_heads.box_predictor.test_score_thresh,
                             model.roi_heads.box_predictor.test_topk_per_image,
                             model.roi_heads.box_predictor.test_nms_thresh))

            if cfg.MODEL.BACKBONE.FREEZE_AT > 0:
                if "vitdet" in cfg.MODEL.BACKBONE.NAME:
                    model.backbone.net.requires_grad_(False) # ViT is a little different from other models.
                else:
                    model.backbone.bottom_up.requires_grad_(False)
            elif cfg.MODEL.BACKBONE.FREEZE_AT > 1:
                model.backbone.requires_grad_(False) # Freeze backbone.
        else:
            model = super().build_model(cfg)

        model.to("cuda")
        logger.info("Model Parameters:\n%s", parameter_count_table(model, max_depth=2))
        return model

    @classmethod
    def build_test_loader(cls, cfg, dataset_name):
        if cfg.MODEL.BACKBONE.NAME in LazyConfigurations:
            # Using mainly the old settings here to support multiple datasets.
            test_mapper_config = Trainer.GetLazyConfig(cfg).dataloader.test.mapper
            test_mapper = instantiate(test_mapper_config)
            return build_detection_test_loader(cfg, dataset_name, mapper=test_mapper, batch_size=cfg.TEST_BATCH_SIZE) # pylint: disable=too-many-function-args
        else:
            return super().build_test_loader(cfg, dataset_name)

    @classmethod
    def build_train_loader(cls, cfg):
        if cfg.MODEL.BACKBONE.NAME in LazyConfigurations:
            logger = logging.getLogger("detectron2.trainer")
            logger.info("Configuring %s train data loader.", cfg.MODEL.BACKBONE.NAME)
            train_loader_config = Trainer.GetLazyConfig(cfg).dataloader.train
            train_loader_config.dataset = L(get_detection_dataset_dicts)(names=cfg.DATASETS.TRAIN)
            train_loader_config.total_batch_size = cfg.SOLVER.IMS_PER_BATCH
            return instantiate(train_loader_config)
        else:
            return super().build_train_loader(cfg)

    @classmethod
    def build_optimizer(cls, cfg, model):
        if cfg.MODEL.BACKBONE.NAME in LazyConfigurations:
            logger = logging.getLogger("detectron2.trainer")
            logger.info("Configuring %s optimizer.", cfg.MODEL.BACKBONE.NAME)
            optimizer_config = model_zoo.get_config("common/optim.py").AdamW
            #optimizer.params.overrides = {"pos_embed": {"weight_decay": 0.0}}
            optimizer_config.lr = cfg.SOLVER.BASE_LR
            optimizer_config.params.model = model
            return instantiate(optimizer_config)
        else:
            return super().build_optimizer(cfg, model)

    @classmethod
    def build_lr_scheduler(cls, cfg, optimizer):
        if cfg.MODEL.BACKBONE.NAME in LazyConfigurations:
            logger = logging.getLogger("detectron2.trainer")
            logger.info("Configuring %s learning rate scheduler.", cfg.MODEL.BACKBONE.NAME)
            scheduler_config = L(MultiStepParamScheduler)(
                values=[1.0, 0.1, 0.01],
                milestones=cfg.SOLVER.STEPS,
                num_updates=cfg.SOLVER.MAX_ITER,
            )
            return instantiate(scheduler_config)
        else:
            return super().build_lr_scheduler(cfg, optimizer)

    def build_hooks(self):
        cfg = self.cfg.clone()
        hooks = super().build_hooks() # pylint: disable=redefined-outer-name
        checkpointer = DetectionCheckpointer(
            self.model, # pylint: disable=no-member
            cfg.OUTPUT_DIR,
        )
        hooks.insert(-1, BestCheckpointer(cfg.TEST.EVAL_PERIOD, checkpointer, "drone_fixed_wing_val_v2/bbox/APs", "max",))
        return hooks

    @staticmethod
    def GetLazyConfig(cfg):
        return LazyConfig.load(LazyConfigurations[cfg.MODEL.BACKBONE.NAME])

    @staticmethod
    def SetupLazy(cfg):
        lazyConfig = Trainer.GetLazyConfig(cfg)
        cfg.INPUT.FORMAT = "RGB"

        cfg.INPUT.MIN_SIZE_TEST = lazyConfig.train.IMAGE_SIZE
        cfg.INPUT.MAX_SIZE_TEST = lazyConfig.train.IMAGE_SIZE

        return Trainer.SetupCommon(cfg)

    @staticmethod
    def Setup(cfg):
        # cfg.INPUT.MIN_SIZE_TEST = 512
        # cfg.INPUT.MAX_SIZE_TEST = 512
        # cfg.INPUT.MIN_SIZE_TRAIN = (256, 384, 512, 640, 768)
        # cfg.MODEL.ANCHOR_GENERATOR.SIZES = [[32], [64], [128], [256]]
        cfg.MODEL.ANCHOR_GENERATOR.ASPECT_RATIOS = [[0.5, 1.0, 2.0]]

        cfg.SOLVER.MAX_ITER = 2000
        cfg.SOLVER.STEPS = (1000,1800)
        cfg.SOLVER.BASE_LR = 0.0005
        cfg.SOLVER.WARMUP_ITERS = 40
        cfg.TEST.EVAL_PERIOD = 200
        #cfg.SOLVER.CHECKPOINT_PERIOD=250

        return Trainer.SetupCommon(cfg)

    @staticmethod
    def SetupCommon(cfg):
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 512

        cfg.MODEL.RPN.IOU_THRESHOLDS = [0.3, 0.7]
        cfg.MODEL.RPN.IOU_LABELS = [0, -1, 1]

        cfg.MODEL.RPN.PRE_NMS_TOPK_TRAIN  = 2000
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST   = 1000
        cfg.MODEL.RPN.POST_NMS_TOPK_TRAIN = 1000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST  = 1000

        cfg.TEST.DETECTIONS_PER_IMAGE = 1000

        cfg.INPUT.MASK_ON = False
        # cfg.INPUT.MASK_FORMAT='bitmask'
        #cfg.DATASETS.TRAIN = (TRAINING_DATASET,)
        #cfg.DATASETS.TEST = (VALIDATION_DATASET,)

        cfg.OUTPUT_DIR = os.path.join( cfg.OUTPUT_DIR, datetime.now().strftime("%Y%m%d-%H%M%S"))

        os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)

        cfg.TEST_BATCH_SIZE = cfg.SOLVER.IMS_PER_BATCH
        cfg.TEST_BATCH_SIZE = max(cfg.TEST_BATCH_SIZE, 1)

        cfg.TEST.AUG.ENABLED = False
        return cfg

    @staticmethod
    def make_config(config_file):
        cfg = get_cfg()
        cfg.merge_from_file(config_file)

        if cfg.MODEL.BACKBONE.NAME in LazyConfigurations:
            Trainer.SetupLazy(cfg)
        else:
            Trainer.Setup(cfg)

        return cfg

    @staticmethod
    def setup(arguments):
        cfg = Trainer.make_config(arguments.config_file)
        cfg.merge_from_list(arguments.opts)
        cfg.freeze()
        default_setup(cfg, arguments)
        return cfg
