# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
"""
Implements the Generalized R-CNN framework
"""

from collections import OrderedDict
import torch
import torch_xla.core.xla_model as xm
from torch import nn


class GeneralizedRCNN(nn.Module):
    """
    Main class for Generalized R-CNN.

    Arguments:
        backbone (nn.Module):
        rpn (nn.Module):
        heads (nn.Module): takes the features + the proposals from the RPN and computes
            detections / masks from it.
        transform (nn.Module): performs the data transformation from the inputs to feed into
            the model
    """

    def __init__(self, backbone, rpn, roi_heads, transform):
        super(GeneralizedRCNN, self).__init__()
        self.transform = transform
        self.backbone = backbone
        self.rpn = rpn
        self.roi_heads = roi_heads

    def forward(self, images, targets=None):
        """
        Arguments:
            images (list[Tensor]): images to be processed
            targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

        Returns:
            result (list[BoxList] or dict[Tensor]): the output from the model.
                During training, it returns a dict[Tensor] which contains the losses.
                During testing, it returns list[BoxList] contains additional fields
                like `scores`, `labels` and `mask` (for Mask R-CNN models).

        """
        if self.training and targets is None:
            raise ValueError("In training mode, targets should be passed")
        original_image_sizes = [img.shape[-2:] for img in images]
        images, targets = self.transform(images, targets)
        features = self.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([(0, features)])
        proposals, proposal_losses = self.rpn(images, features, targets)
        print("models/detection/generalized_rcnn.py - RoIHeads.forward(...) start")
        detections, detector_losses = self.roi_heads(features, proposals, images.image_sizes, targets)
        print("models/detection/generalized_rcnn.py - RoIHeads.forward(...) end")
        print("models/detection/generalized_rcnn.py - GeneralizedRCNNTransform.postprocess(...) start")

        if not self.training:
            # Sync tensors and use CPU instead for post processing
            sync_tensors = []
            for detection in detections:
                sync_tensors.extend(list(detection.values()))
            xm.mark_step()

            detections_cpu = []
            for detection in detections:
                detection_cpu = {}
                for k in detection.keys():
                    detection_cpu[k] = detection[k].cpu().clone()
                detections_cpu.append(detection_cpu)
            detections = detections_cpu

        detections = self.transform.postprocess(detections_cpu, images.image_sizes, original_image_sizes)
        print("models/detection/generalized_rcnn.py - GeneralizedRCNNTransform.postprocess(...) end")

        losses = {}
        losses.update(detector_losses)
        losses.update(proposal_losses)

        if self.training:
            return losses

        return detections
