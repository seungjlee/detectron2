from detectron2.config import configurable
from detectron2.modeling.roi_heads.cascade_rcnn import CascadeROIHeads


#@ROI_HEADS_REGISTRY.register()
class CascadeROIHeadsPlus(CascadeROIHeads):
    """
    Customized CascadeROIHeads.
    """
    @configurable
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def forward(self, images, features, proposals, targets=None):
        predictions, losses = super().forward(images, features, proposals, targets)
        return predictions, losses
