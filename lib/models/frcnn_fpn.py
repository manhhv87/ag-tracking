import torch
import torch.nn.functional as F
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.transform import resize_boxes
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone


class FRCNN_FPN(FasterRCNN):
    """
    Faster R-CNN with an FPN backbone and a replaceable classification head.

    Parameters
    ----------
    backbone_type : str, default "ResNet50FPN"
        Which ResNet-FPN backbone to use. Supported in this file:
        "ResNet34FPN", "ResNet50FPN", "ResNet101FPN".
    num_classes : int, default 2
        Number of classes **including background** as required by
        `FastRCNNPredictor` / torchvision detectors.
    pretrained : bool, default False
        Whether to initialize ResNet weights with pretrained ImageNet weights.
        (Note: For the ResNet-50 case, the code uses the `weights=None` form.)
    """

    def __init__(self, backbone_type="ResNet50FPN", num_classes=2, pretrained=False):
        print(f"use backbone: {backbone_type}")

        # Select and build an FPN backbone from torchvision based on the name.
        if backbone_type == "ResNet34FPN":
            backbone = resnet_fpn_backbone("resnet34", pretrained)
        if backbone_type == "ResNet50FPN":
            # Uses the explicit keyword `backbone_name` and disables pretrained weights
            # via `weights=None` (consistent with newer torchvision API).
            backbone = resnet_fpn_backbone(backbone_name="resnet50", weights=None)
        if backbone_type == "ResNet101FPN":
            backbone = resnet_fpn_backbone("resnet101", pretrained)
        
        # Initialize the parent FasterRCNN with the chosen backbone and num_classes.
        super(FRCNN_FPN, self).__init__(backbone, num_classes)

        # Replace the classification/regression head so its output matches `num_classes`.
        in_features = self.roi_heads.box_predictor.cls_score.in_features
        self.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

        # Adjust the RoIHeads NMS threshold (default is typically 0.5; set explicitly).
        self.roi_heads.nms_thresh = 0.5

    def detect_batches(self, images):
        """
        Run detection on a batch of images and also expose intermediate features.

        Parameters
        ----------
        images : torch.Tensor
            A 4D batch tensor of shape (N, C, H, W). This follows the
            torchvision detection API when `FasterRCNN.transform` is
            applied internally.

        Returns
        -------
        detections : List[Dict[str, Tensor]]
            Per-image predictions (boxes, labels, scores) after resizing back
            to original image sizes.
        features : Dict[str, Tensor] or Tensor
            Backbone feature maps as produced by the FPN.
        preprocessed_image_sizes : List[Tuple[int, int]]
            Spatial sizes (H, W) for each image **after** internal transforms.
        """
        # Use the device of the model's parameters to place the inputs.
        device = list(self.parameters())[0].device
        images = images.to(device)

        # Save original image sizes (H, W) for later postprocess scaling.
        self.original_image_sizes = [img.shape[-2:] for img in images]

        # Apply the same transforms as in the standard Faster R-CNN forward pass.
        # `images` becomes an ImageList (with .tensors and .image_sizes); targets=None.
        images, _ = self.transform(images, None)
        self.preprocessed_image_sizes = images.image_sizes
        
        # 1) Backbone feature extraction (FPN returns a dict of feature maps).
        features = self.backbone(images.tensors)  # dict('1-4, pool'), tensors inside
        
        # 2) Region Proposal Network (RPN): propose candidate boxes.
        # Returns a list of proposal tensors per image (each of shape [num_props, 4]).
        proposals, _ = self.rpn(
            images, features
        )  # Meshgrid Warning comes from here, (list: 10, Tensors(1000, 4))1
        
        # 3) ROI heads: classification + box regression on pooled features.
        detections, _ = self.roi_heads(
            features, proposals, self.preprocessed_image_sizes
        )

        # 4) Post-processing: resize detections back to original image sizes.
        detections = self.transform.postprocess(
            detections, self.preprocessed_image_sizes, self.original_image_sizes
        )

        # Return postprocessed detections, raw features, and the preprocessed sizes.
        return detections, features, self.preprocessed_image_sizes

    def predict_boxes(self, boxes, features, class_id=1):
        """Regress only single image for tracktor"""
        # NOTE: Retaining the original one-line docstring from the code to avoid
        # any behavioral changes. Detailed explanation follows in comments.

        # Use the model's device for all downstream tensors.
        device = list(self.parameters())[0].device
        boxes = boxes.to(device)

        # Resize incoming boxes from original image size to the preprocessed size
        # (the size used by the internal transform and ROI operations).
        boxes = resize_boxes(
            boxes, self.original_image_sizes[0], self.preprocessed_image_sizes[0]
        )

        # The ROI heads expect a *list* of proposals per image, so wrap in a list.
        proposals = [boxes]

        # Pool ROI-aligned features for the proposals on the given feature maps.
        box_features = self.roi_heads.box_roi_pool(
            features, proposals, [self.preprocessed_image_sizes[0]]
        )

        # Pass pooled features through the box head (typically 2 FC layers).
        box_features = self.roi_heads.box_head(box_features)

        # Predict class logits and encoded box deltas for each ROI.
        class_logits, box_regression = self.roi_heads.box_predictor(box_features)

        # Decode the regressed boxes w.r.t. the proposals using the RoI box coder.
        pred_boxes = self.roi_heads.box_coder.decode(box_regression, proposals)

        # Convert class logits into per-class probabilities.
        pred_scores = F.softmax(class_logits, -1)

        # Select the set of boxes and scores for the requested `class_id`.
        # Shape of pred_boxes is [N, num_classes, 4]; index into class dim.
        pred_boxes = pred_boxes[:, class_id].detach()
        pred_scores = pred_scores[:, class_id].detach()

        # Resize the predicted boxes back from preprocessed size to original size.
        pred_boxes = resize_boxes(
            pred_boxes, self.preprocessed_image_sizes[0], self.original_image_sizes[0]
        )

        # Return per-ROI boxes and scores for the specified class.
        return pred_boxes, pred_scores
