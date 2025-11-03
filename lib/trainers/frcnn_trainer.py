import os
import sys
import cv2
import tqdm
import torch
import shutil
import datetime
import numpy as np
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

from ..utils.det_utils import utils
from ..models.frcnn_fpn import FRCNN_FPN
from ..utils.det_utils.engine import train_one_epoch, evaluate


class FrcnnTrainer:
    """
    Thin orchestration class for training/evaluating a Faster R-CNN + FPN model.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary. Expected keys include (non-exhaustive):
        - "mode": str, either "train" or something else (controls run dir creation).
        - "outdir": str, base output directory for logs/artifacts.
        - "backbone": str, backbone type passed to `FRCNN_FPN`.
        - "batch_size": int, DataLoader batch size.
        - "num_workers": int, DataLoader workers.
        - "lr", "momentum", "weight_decay": SGD hyperparameters.
        - "step_size", "gamma": StepLR scheduler parameters.
        - "epoch_start": int, starting epoch index (useful when resuming).
        - "epochs": int, total number of epochs to train.
        - "checkpoint": str or empty/False, path to a model state dict to load.
    dset : torch.utils.data.Dataset
        Detection dataset compatible with torchvision-style detectors.
        Each __getitem__ should return (images, targets) where images is a tuple/list
        and targets is a tuple/list of dicts containing keys like "boxes", "labels",
        and for `predict()`, a preview image in `targets[0]["img"]` (C,H,W).
    """

    def __init__(self, cfg, dset):
        # Store raw config; downstream methods read values directly as needed.
        self.cfg = cfg

        # If training mode, create a timestamped run directory under outdir.
        if cfg["mode"] == "train":
            path = cfg["outdir"] + "det/frcnn/train/"
            # Timestamp folder: YY.MM.DD.HH.MM
            path += datetime.datetime.now().strftime("%y.%m.%d.%H.%M")
            # Copy the complete config folder into the run directory for traceability.
            shutil.copytree("config", path + "/config")
            self.save_path = path
            # Store weights under a subfolder named after the chosen backbone.
            self.weights_path = path + "/%s_weights" % cfg["backbone"]
        
        # Keep a reference to the dataset; used by the DataLoader below.
        self.dataset = dset

        # Training schedule
        self.epoch_start = cfg["epoch_start"]
        self.num_epochs = cfg["epochs"]

        # DataLoader with custom collate function suitable for detection targets.
        self.loader = DataLoader(
            dset,
            batch_size=cfg["batch_size"],
            shuffle=True,
            num_workers=cfg["num_workers"],
            collate_fn=utils.collate_fn,
        )

        # Instantiate the detection model with the requested backbone.
        self.model = FRCNN_FPN(backbone_type=cfg["backbone"])

        # Optional checkpoint loading (resuming from previous training).
        if self.cfg["checkpoint"]:
            print(
                'loading checkpoint %d: "%s"' % (cfg["epoch_start"], cfg["checkpoint"])
            )
            model_state_dict = torch.load(cfg["checkpoint"])
            self.model.load_state_dict(model_state_dict)

        # Optimizer: classic SGD setup for detection models.
        self.optimizer = torch.optim.SGD(
            self.model.parameters(),
            lr=cfg["lr"],
            momentum=cfg["momentum"],
            weight_decay=cfg["weight_decay"],
        )
        
        # Step-wise LR decay: reduce LR every `step_size` epochs by factor `gamma`.
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=cfg["step_size"], gamma=cfg["gamma"]
        )
        
        # For simplicity, training is pinned to a single GPU device "cuda:0".
        # (Adapt/parametrize this if running multi-GPU/DDP.)
        self.device = torch.device("cuda:0")

    def train(self):
        """
        Run the full multi-epoch training loop.

        Behavior
        --------
        - Creates the weights directory if it does not exist.
        - Redirects stdout/stderr to `train.log` under the run directory.
        - Moves the model to GPU.
        - For each epoch:
            * prints epoch header,
            * calls `train_one_epoch` with progress logging,
            * every 10 epochs, saves a checkpoint and runs `evaluate` on the
              training set (as provided by `self.loader`).
        - Flushes the log stream after each epoch.

        Notes
        -----
        - This routine evaluates on the same training loader for quick feedback.
          For a true validation score, replace `self.loader` with a validation
          dataloader when calling `evaluate`.
        """
        # Ensure the checkpoint directory exists.
        os.makedirs(self.weights_path, exist_ok=True)

        # Redirect all prints to a log file for reproducibility and later review.
        sys.stdout = open(self.save_path + "/train.log", "w")
        sys.stderr = sys.stdout

        # Move model to the selected device once; dataloader tensors are moved later.
        self.model.to(self.device)

        # Epoch-major training loop
        for epoch in range(self.epoch_start, self.num_epochs + 1):
            print("\n++++++++++++++++++++")
            print(">>> Epoch %03d ++++++" % (epoch))
            print("++++++++++++++++++++")

            # One full epoch of training with periodic metric printing.
            train_one_epoch(
                self.model,
                self.optimizer,
                self.loader,
                self.device,
                epoch,
                print_freq=10,
            )

            # Every 10 epochs, persist weights and run a quick on-train evaluation.
            if epoch % 10 == 0:
                torch.save(
                    self.model.state_dict(),
                    self.weights_path + "/model_epoch_{}.pt".format(epoch),
                )

                # NOTE: Evaluates on the training set for speed/visibility.
                evaluate(
                    self.model, self.loader, self.device
                )
            
            # Ensure logs are written to disk promptly (useful on long runs).
            sys.stdout.flush()

    def eval(self):
        """
        Evaluate the current model on the configured loader and print COCO stats.

        Behavior
        --------
        - If a checkpoint path is provided in the config, it prints which one is used.
          Otherwise, it warns that random weights are being evaluated.
        - Runs `evaluate` and prints a selection of COCO metrics with short labels.

        Output
        ------
        Prints the values returned by `CocoEvaluator` (APs at different IoU/scale and AR).
        """
        if self.cfg["checkpoint"]:
            print("Using checkpoint:", self.cfg["checkpoint"])
        else:
            print("WARNING: No checkpoint provided. Using random weights.")

        # Short labels matching COCO evaluator's `stats` ordering
        stat_description = [
            "APa",
            "AP.5",
            "AP.75",
            "APs",
            "APm",
            "APl",
            "AR1",
            "AR10",
            "AR100",
            "APs",
            "APm",
            "APl",
        ]

        # Print a small header before the full results
        print(stat_description[0], stat_description[8])

        # Move model to device and run the standard evaluation.
        self.model.to(self.device)
        evaluator = evaluate(self.model, self.loader, self.device)

        # CocoEvaluator stores results in `coco_eval` dict keyed by IoU type.
        # We take the first (typically "bbox") and read its `.stats` vector.
        stats = list(evaluator.coco_eval.values())[0].stats

        # Pair each short label with its value; mark a couple for emphasis.
        for desc, stat in zip(stat_description, stats):
            print(desc, stat, "*" if desc in ["APa", "AR100"] else "")

    def predict(self, img=None):
        """
        Produce quick qualitative predictions and dump them as images.

        Parameters
        ----------
        img : optional
            If provided, should be a **single image tensor** (C,H,W) already on
            the correct scale and dtype for the model. When `img` is given:
              - the model is called directly on this tensor,
              - the routine prints the shape of the returned result tensor/dict,
              - visualization is marked TODO and the function returns early.

            If `img` is None:
              - the method iterates over `self.loader` and performs single-image
                inference per batch element (using only `imgs[0]` for brevity),
              - draws bounding boxes and confidence scores,
              - shows the image via `cv2.imshow` and writes it to disk,
              - accumulates confidences and saves a histogram at the end.

        Notes
        -----
        - The visualization expects `target[0]["img"]` to be present and to
          contain the original image as a tensor with channel order (B,G,R)
          accessible as `img[0], img[1], img[2]` respectively.
        - This visualization is intended for quick inspection, not for
          production-grade rendering.
        """
        # Create a timestamped directory for prediction artifacts.
        path = self.cfg["outdir"] + "det/frcnn/predict/"
        path += datetime.datetime.now().strftime("%y.%m.%d.%H.%M")
        os.makedirs(path, exist_ok=True)

        # Switch to eval mode to disable dropout/BN updates and reduce overhead.
        self.model.to(self.device)
        self.model.eval()

        # Fast path: run inference on a user-provided image tensor.
        if img is not None:
            res = self.model(img)
            print(res.shape)
            # TODO: Implement the visualization of the results
            # Save in det/predict/date/name.jpg
            return
        
        # Otherwise, iterate over the dataloader and visualize predictions.
        confs = np.array([])    # store all confidence scores to histogram later
        with torch.no_grad():
            for i, (imgs, target) in tqdm.tqdm(
                enumerate(self.loader), desc="Predicting"
            ):
                # Take the first image from the batch and move to device.
                imgs = imgs[0].to(self.device)

                # The detector expects a list of images; take single-image mode.
                out = self.model([imgs])[0]

                # Convert predicted boxes to integer pixel coords for drawing.
                boxes = out["boxes"].cpu().numpy().astype(int)
                # Accumulate confidences for the histogram.
                confs = np.concatenate([confs, out["scores"].cpu().numpy()])

                # Retrieve the original (BGR) image from the target; convert to HxWxC.
                img = target[0]["img"].cpu().numpy()
                B, G, R = img[0, :, :], img[1, :, :], img[2, :, :]
                img = np.stack([B, G, R], 2)

                # Draw each predicted box and its score.
                for bb, sc in zip(boxes, out["scores"].cpu().numpy()):
                    x1, y1, x2, y2 = bb
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 1), 2)
                    cv2.putText(
                        img,
                        "%.2f" % sc,
                        (x1, y2 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1,
                        (1, 0, 0),
                        2,
                    )

                # Show interactive preview and persist to disk (scaled to [0,255]).
                cv2.imshow("video", img)
                cv2.imwrite(path + "/%03d.jpg" % i, (img * 255).astype(np.uint8))
                cv2.waitKey(1)

                # Limit the preview to the first 11 items for convenience.
                if i == 10:
                    break
            
        # Plot and save a histogram of all confidence scores.
        plt.figure()
        plt.hist(confs)
        plt.savefig(path + "/000.conf.hist.jpg")
