import detectron2
import detectron2.config
import numpy
import torch
from detectron2.modeling.roi_heads.fast_rcnn import fast_rcnn_inference_single_image
from itertools import count
from torchvision.transforms import Resize, InterpolationMode


def MergeDetections(all_boxes, all_scores, all_classes, shape_hw, cfg: detectron2.config.CfgNode):
    # select from the union of all results
    num_boxes = len(all_boxes)
    num_classes = cfg.MODEL.ROI_HEADS.NUM_CLASSES
    # +1 because fast_rcnn_inference expects background scores as well
    all_scores_2d = torch.zeros(num_boxes, num_classes + 1, device=all_boxes.device)
    for idx, cls, score in zip(count(), all_classes, all_scores):
        all_scores_2d[idx, cls] = score

    merged_instances, _ = fast_rcnn_inference_single_image(
        all_boxes,
        all_scores_2d,
        shape_hw,
        1e-8,
        cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST,
        cfg.TEST.DETECTIONS_PER_IMAGE,
    )

    return merged_instances

def MergeInstances(instances1, instances2, shape_hw, cfg):
    pred_boxes = torch.cat((instances1.pred_boxes.tensor, instances2.pred_boxes.tensor))
    pred_classes = torch.cat((instances1.pred_classes, instances2.pred_classes))
    pred_scores = torch.cat((instances1.scores, instances2.scores))
    return MergeDetections(pred_boxes, pred_scores, pred_classes, shape_hw, cfg)

def BatchedInference(detection_model, input_images):
    with torch.no_grad():
        inputs = []
        for image in input_images:
            height, width = image.shape[1:3]
            inputs.append({"image": image, "height": height, "width": width})

        return detection_model(inputs)

class InferenceModel(torch.nn.Module):
    def __init__(self, cfg: detectron2.config.CfgNode, detection_model: torch.nn.Module,
                 dtype: torch.dtype, device: torch.device,
                 net_input_size: tuple, window_size: int,
                 window_rows: int, window_cols: int,
                 center_crop_size: tuple = None,
                 border: int = 2):
        super().__init__()
        self.cfg = cfg
        self.model = detection_model
        self.net_input_size = net_input_size
        self.dtype = dtype
        self.device = device
        self.window_size = window_size
        self.rows = window_rows
        self.cols = window_cols
        self.center_crop_size = center_crop_size
        self.border = border

        self.sampling_factor = self.window_size / self.net_input_size

        self.resize_to_net = Resize((self.net_input_size,self.net_input_size),
                                    interpolation=InterpolationMode.BILINEAR, antialias=False)

    def forward(self, img: numpy.ndarray):
        offset_y, offset_x = 0, 0
        size_y, size_x = img.shape[:2]
        if self.center_crop_size:
            size_y, size_x = self.center_crop_size
            assert img.shape[0] >= size_y
            assert img.shape[1] >= size_x
            offset_y = (img.shape[0] - size_y) // 2
            offset_x = (img.shape[1] - size_x) // 2

        step_y = (size_y - self.window_size) // (self.rows-1) if self.rows > 1 else 0
        step_x = (size_x - self.window_size) // (self.cols-1) if self.cols > 1 else 0

        scan_windows = [
            (offset_y + step_y*y, offset_x + step_x*x,
             offset_y + step_y*y + self.window_size, offset_x + step_x*x + self.window_size)
            for x in range(self.cols) for y in range(self.rows)
            if step_y*y+self.window_size <= size_y and step_x*x+self.window_size <= size_x
        ]

        pred_boxes = [None] * len(scan_windows)
        pred_classes = [None] * len(scan_windows)
        pred_scores = [None] * len(scan_windows)

        image_size = img.shape[:2]
        img_tensor = torch.from_numpy(img.transpose(2,0,1)).to(dtype=self.dtype, device=self.device)

        subimages = [
            self.resize_to_net(img_tensor[:,subwindow[0]:subwindow[2],subwindow[1]:subwindow[3]])
            for subwindow in scan_windows
        ]
        inference_outputs = BatchedInference(self.model, subimages)
        for idx, outputs in enumerate(inference_outputs):
            pred_boxes[idx] = outputs['instances'].pred_boxes.tensor
            pred_classes[idx] = outputs['instances'].pred_classes
            pred_scores[idx] = outputs['instances'].scores

            # Filter out boxes that are too close to winodw edge.
            good = (
                (pred_boxes[idx][:,0] >= self.border) &
                (pred_boxes[idx][:,1] >= self.border) &
                (pred_boxes[idx][:,2] < self.window_size-self.border) &
                (pred_boxes[idx][:,3] < self.window_size-self.border)
            )
            pred_boxes[idx] = pred_boxes[idx][good]
            pred_classes[idx] = pred_classes[idx][good]
            pred_scores[idx] = pred_scores[idx][good]

            pred_boxes[idx] *= self.sampling_factor

            pred_boxes[idx][:,0] += torch.tensor(scan_windows[idx][1], device=pred_boxes[idx].device)
            pred_boxes[idx][:,1] += torch.tensor(scan_windows[idx][0], device=pred_boxes[idx].device)
            pred_boxes[idx][:,2] += torch.tensor(scan_windows[idx][1], device=pred_boxes[idx].device)
            pred_boxes[idx][:,3] += torch.tensor(scan_windows[idx][0], device=pred_boxes[idx].device)

        pred_boxes = torch.cat(pred_boxes, dim=0)
        pred_classes = torch.cat(pred_classes, dim=0)
        pred_scores = torch.cat(pred_scores, dim=0)

        # 80 = drone 4-axis
        # 81 = drone fixed-wing
        drones = (pred_classes == 80) | (pred_classes == 81)
        pred_boxes = pred_boxes[drones]
        pred_classes = pred_classes[drones]
        pred_scores = pred_scores[drones]

        outputs = MergeDetections(pred_boxes, pred_scores, pred_classes, image_size, self.cfg)
        return outputs, scan_windows

    def simple_inference(self, original_image, preprocess):
        with torch.no_grad():
            height, width = original_image.shape[:2]
            transforms = preprocess(detectron2.data.transforms.AugInput(original_image))
            image = transforms.apply_image(original_image)

            inputs = {"image": torch.tensor(image.transpose(2,0,1)), "height": height, "width": width}
            outputs = self.model([inputs])[0]
            return outputs['instances']
