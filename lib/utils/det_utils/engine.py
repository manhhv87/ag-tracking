import sys
import math
import time

import torch
import torchvision

from . import utils
from .coco_eval import CocoEvaluator
from .coco_utils import get_coco_api_from_dataset


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq):
    """
    Run one full training epoch.

    Parameters
    ----------
    model : torch.nn.Module
        Detection model (e.g., Faster/Mask R-CNN). In training mode it should
        accept a list of image tensors and a list of target dicts, and return
        a dict of losses (scalars as tensors).
    optimizer : torch.optim.Optimizer
        Optimizer used to update model parameters.
    data_loader : torch.utils.data.DataLoader
        Iterable yielding `(images, targets)` where `images` is an iterable of
        tensors and `targets` an iterable of dicts with annotations.
    device : torch.device
        Device for computation (e.g. 'cuda' or 'cpu').
    epoch : int
        Current epoch index; used to optionally enable LR warmup at epoch 0.
    print_freq : int
        How many iterations between metric printouts.

    Behavior
    --------
    - Puts the model into training mode (`model.train()`).
    - Creates a `MetricLogger` with an LR meter (window_size=1 for instant LR).
    - (Common pattern) If this is the first epoch, builds a short warmup LR
      scheduler for a stable ramp-up of the learning rate.
    - Sends images and targets to the `device`, computes losses, backpropagates,
      steps the optimizer (and the warmup scheduler if present), and logs
      smoothed loss statistics and current learning rate.

    Notes
    -----
    - If the reduced loss becomes non-finite, training is aborted to prevent
      undefined behavior.
    """
    model.train()  # ensure layers like dropout/batchnorm are in training mode
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", utils.SmoothedValue(window_size=1, fmt="{value:.6f}"))

    # The header string is used by MetricLogger to prefix log lines.
    header = "Epoch: [{}]".format(epoch)

    # Warmup LR only for the very first epoch, a common stabilization trick.
    if epoch == 0:
        # start at 0.1% of the base LR
        warmup_factor = 1.0 / 1000
        # Cap warmup steps by the loader length; guard against tiny datasets.
        warmup_iters = min(1000, len(data_loader) - 1)
        # Build a LambdaLR that linearly increases LR to 1.0 over warmup_iters.
        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    else:
        lr_scheduler = None

    # Wrap the data_loader so MetricLogger can measure timings and print periodically.
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        # Move each image tensor to the computation device.
        images = list(image.to(device) for image in images)
        # Move every field inside each target dict to the device as well.
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Forward pass (training signature): returns a dict of individual losses.
        loss_dict = model(images, targets)

        # Sum all loss terms (typical for detection models) into a single scalar.
        losses = sum(loss for loss in loss_dict.values())

        # For logging on multi-GPU, reduce losses across ranks to consistent values.
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        # Abort training if loss exploded or became NaN/Inf.
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        # Standard optimization step: clear grads -> backward -> optimizer step.
        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        # Advance warmup schedule (if it was created for epoch 0).
        if lr_scheduler is not None:
            lr_scheduler.step()

        # Update the smoothed logger with the (reduced) loss values.
        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        # Record the current LR (first param group; typical single-LR setup).
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])


def _get_iou_types(model):
    """
    Infer which IoU types are applicable for evaluation based on the model class.

    Parameters
    ----------
    model : torch.nn.Module or torch.nn.parallel.DistributedDataParallel
        The detection model. If wrapped in DDP, we inspect `model.module`.

    Returns
    -------
    list[str]
        Always includes "bbox". Adds "segm" for Mask R-CNN and "keypoints" for
        Keypoint R-CNN models.
    """
    # Unwrap the underlying module if the model is wrapped with DDP.
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    
    # Start with bounding-box IoU evaluation.
    iou_types = ["bbox"]

    # If the model is Mask R-CNN, also evaluate mask IoU ("segm").
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    
    # If the model is Keypoint R-CNN, also evaluate keypoint IoU ("keypoints").
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")

    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device):
    """
    Evaluate the model using COCO metrics via `CocoEvaluator`.

    Parameters
    ----------
    model : torch.nn.Module
        In eval mode, should accept a list of images and return a list of
        prediction dicts (one per image).
    data_loader : torch.utils.data.DataLoader
        Iterable yielding `(images, targets)`; targets contain `image_id` and
        other fields required for COCO-format results.
    device : torch.device
        Device to run inference on.

    Returns
    -------
    CocoEvaluator
        The evaluator after `accumulate()` and `summarize()` have been called.

    Procedure
    ---------
    - Limit Torch’s CPU threads to 1 during evaluation (mitigates contention).
    - Ensure `model.eval()` and disable grad with `@torch.no_grad()`.
    - Build the COCO API object from the dataset and derive applicable IoU types.
    - For each batch:
        * move images/targets to the device,
        * synchronize and time the forward pass,
        * move outputs back to CPU for COCO processing,
        * create {image_id: prediction} mapping and update evaluator,
        * log per-batch model/evaluator timings.
    - Synchronize evaluator and logger across processes,
      then accumulate and summarize metrics.
    """
    # Remember original thread count then temporarily force 1 thread for eval.
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)

    # We keep a CPU device handle for moving predictions back (COCO API expects CPU tensors).
    cpu_device = torch.device("cpu")
    
    # Switch to evaluation mode (affects dropout/batchnorm behavior).
    model.eval()

    # Metric logger to time model inference and evaluator updates.
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = "Test:"

    # Obtain the COCO API from the dataset; needed to construct CocoEvaluator.
    coco = get_coco_api_from_dataset(data_loader.dataset)

    # Determine which IoU types we should evaluate (bbox/segm/keypoints).
    iou_types = _get_iou_types(model)

    # Evaluator that accumulates predictions and computes COCO metrics.
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # Iterate over evaluation data and periodically print averaged timings.
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        # Move images and targets to the evaluation device.
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # Synchronize to get accurate timing for the model's forward pass.
        torch.cuda.synchronize()
        model_time = time.time()

        # Inference: returns a list of predictions (dict) per input image.
        outputs = model(image)

        # Move predictions to CPU for COCO formatting/computation.
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time   # elapsed model (inference) time

        # Map each prediction to its corresponding image id from targets.
        res = {
            target["image_id"].item(): output
            for target, output in zip(targets, outputs)
        }

        # Time the evaluator update (formatting + adding results).
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time

        # Log per-batch timings for visibility and performance tracking.
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator
