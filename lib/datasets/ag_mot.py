import os
import cv2
import kornia
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset

from ..utils.data_utils import get_mot_gt
from ..utils.det_utils.utils import get_transform


class AgMOT(Dataset):
    """Agricultural MOT dataset for a single sequence.

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with at least:
        - cfg["datadir"]: base directory pointing to sequence folders.
          Expected structure: `<datadir>/<name>/img/` and `<datadir>/<name>/gt/gt.txt`.
    name : str
        Sequence name (subdirectory under `cfg["datadir"]`).
    train : bool, default False
        If True, the returned annotation tensor (internal) is trimmed to the exact
        number of annotations in the frame (instead of the fixed `anchors` size).
    anchors : int, default 200
        Maximum number of annotations considered per frame. Used to preallocate a
        fixed-size tensor (filled with -1 for missing rows) for convenience.

    Returns (per __getitem__)
    -------------------------
    imgPIL : PIL.Image.Image
        The RGB image as a PIL object (later converted by transforms).
    target : dict
        Dictionary with keys:
          - "img": float tensor from OpenCV/Kornia in range [0, 1], shape (C,H,W).
          - "boxes": tensor of shape (N, 4) with [x1, y1, x2, y2].
          - "ids": tensor of track IDs (same length as boxes).
          - "labels": int64 tensor of class indices (here min(ann_class, 1)).
          - "image_id": scalar tensor with the frame index (0-based).
          - "area": tensor with box areas (w*h).
          - "iscrowd": int64 tensor of zeros of length `anchors`.
    """

    def __init__(self, cfg, name, train=False, anchors=200):
        self.train = train

        # Build the transform pipeline (ToTensor() and optional RandomHorizontalFlip)
        self.transforms = get_transform(self.train)
        self.anchors = anchors
        
        # Read ground-truth bounding boxes from CSV-like gt file
        # Expected format is MOT-style; parser returns a list of rows (lists).
        self.gts = get_mot_gt(cfg["datadir"] + name + "/gt/gt.txt")

        # Image directory and sorted list of image file paths
        self.imgdir = cfg["datadir"] + name + "/img/"
        self.files = sorted([self.imgdir + img for img in os.listdir(self.imgdir)])

    def __len__(self):
        # Number of frames equals number of image files
        return len(self.files)

    def __getitem__(self, index):
        # -----------------------------
        # 1) Load the current frame
        # -----------------------------
        # As PIL (RGB) — used by downstream transforms and models
        imgPIL = Image.open(self.files[index]).convert("RGB")

        # As OpenCV (BGR) tensor via Kornia — auxiliary copy kept in target["img"]
        img = cv2.imread(self.files[index])
        img = kornia.image_to_tensor(img, True).float() / 255.0 # (C,H,W) in [0,1]

        # -----------------------------
        # 2) Select all annotations for this frame (1-based frame index in GT)
        # -----------------------------
        anns = [ann for ann in self.gts if ann[0] == index + 1]
        if anns:
            anns = torch.tensor(anns)
        else:
            # No GT in this frame: use an empty (0,6) tensor
            anns = torch.zeros(0, 6)
        
        # Preallocate a fixed-size annotation tensor filled with -1
        anntensor = -torch.ones(self.anchors, 6)
        # Copy available annotations into the top rows
        anntensor[: len(anns), :] = anns
        
        # If train, return only corresponding bounding boxes
        if self.train:
            anntensor = anntensor[: len(anns), :]
        
        # -----------------------------
        # 3) Build target dictionary
        # -----------------------------
        ids = anntensor[:, 1]           # track IDs (column 1)
        boxes = anntensor[:, -4:]       # last 4 columns are [x1, y1, x2, y2]
        # Map label column 0 into {0,1} by clamping at 1, and cast to int64
        labels = torch.minimum(anntensor[:, 0], torch.tensor(1)).to(dtype=torch.int64)
        image_id = torch.tensor(index)  # 0-based image index

        # Box areas = (x2 - x1) * (y2 - y1); works even if rows are -1 (area negative)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])

        # COCO-style `iscrowd` flag — here all zeros, length = anchors
        iscrowd = torch.zeros(self.anchors, dtype=torch.int64)
        
        target = {
            "img": img,           # keep the original float tensor for visualization
            "boxes": boxes,       # detection boxes (may include -1 rows when not trimmed)
            "ids": ids,           # tracking IDs
            "labels": labels,     # class indices (binary in this setup)
            "image_id": image_id, # frame index (tensor scalar)
            "area": area,         # box areas (w*h)
            "iscrowd": iscrowd,   # all zeros (no crowd instances)
        }

        # -----------------------------
        # 4) Apply transforms (on PIL image + target dict)
        # -----------------------------
        if self.transforms is not None:
            imgPIL, target = self.transforms(imgPIL, target)

        # Return a PIL image tensor (after transforms) and the target dict
        return imgPIL, target


class MergedAgMOT(AgMOT):
    """Concatenate multiple `AgMOT` datasets into a single dataset.

    Parameters
    ----------
    dsets : list[AgMOT]
        A list of prebuilt `AgMOT` datasets to be merged. The resulting dataset
        inherits `train`, `transforms`, and `anchors` from the first element.
        Ground-truth rows are appended with an **offset** to frame indices so that
        images from subsequent datasets have unique `image_id`s.

    Behavior
    --------
    - Concatenate `files` (image paths) from all datasets.
    - For each subsequent dataset, offset the first column (frame index) of its
      `gts` by the number of already known frames (`npre`), then append `gts`.
    """

    def __init__(self, dsets):
        # Inherit common flags/settings from the first dataset
        self.train = dsets[0].train
        self.transforms = get_transform(self.train)
        self.anchors = dsets[0].anchors

        # Initialize with the first dataset's GT and file list
        self.gts = dsets[0].gts
        self.files = dsets[0].files

        # For each remaining dataset, append its files and offset its frame indices
        for dset in dsets[1:]:
            npre = len(self.files)  # number of frames before appending this dataset
            self.files += dset.files
            
            # Adjust every row's image/frame index (column 0) by npre to keep unique IDs
            for trk in dset.gts:
                trk[0] += npre

            # Append ground-truth rows
            self.gts += dset.gts
