import json
import tempfile

import numpy as np
import copy
import time
import torch

try:
    import torch._six as _six  # PyTorch <= 1.9
except Exception:

    class _six:
        string_classes = (str,)
        int_classes = (int,)


from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
import pycocotools.mask as mask_util

from collections import defaultdict

from . import utils


class CocoEvaluator(object):
    """
    Thin wrapper around pycocotools' COCOeval for batched evaluation of detection,
    segmentation, or keypoints—supporting multi-process accumulation/merge.

    Typical usage in a training/validation loop
    -------------------------------------------
    1) Construct once with GT api and iou_types (e.g., ["bbox"]).
    2) For each batch/worker, call `update(predictions)` where `predictions` is a
       dict mapping image_id -> { "boxes"/"scores"/"labels"/... } (Torch tensors).
    3) After distributed sync, call `synchronize_between_processes()` to gather
       shards (img_ids + evalImgs) and build a common `COCOeval` object.
    4) Call `accumulate()` then `summarize()` to print official COCO metrics.

    Why it matters in the paper's pipeline
    --------------------------------------
    Detector quality (mAP here) affects Tracktor-style thresholds (spawn/keep-alive)
    and ultimately ID stability and association in MOT, as discussed in the paper.
    """

    def __init__(self, coco_gt, iou_types):
        # Store a private copy of the ground-truth COCO api to avoid in-place mutations
        assert isinstance(iou_types, (list, tuple))
        coco_gt = copy.deepcopy(coco_gt)
        self.coco_gt = coco_gt

        self.iou_types = iou_types
        self.coco_eval = {}
        for iou_type in iou_types:
            # Build one COCOeval instance per IoU type (bbox/segm/keypoints)
            self.coco_eval[iou_type] = COCOeval(coco_gt, iouType=iou_type)
            # Allow more max detections for large scenes (default [-1] is 100; set to 200)
            self.coco_eval[iou_type].params.maxDets[-1] = 200  # BH: added this line

        self.img_ids = []  # list of image ids seen during updates
        self.eval_imgs = {k: [] for k in iou_types}  # per-iou_type shards of evalImgs

    def update(self, predictions):
        """
        Convert a dict of model predictions into COCO-format results and run
        per-image evaluation (but do not accumulate/summarize yet).

        Parameters
        ----------
        predictions : Dict[int, Dict[str, Tensor]]
            image_id -> { "boxes": [N,4], "scores": [N], "labels": [N],
                           ("masks": [N,1,H,W]) or ("keypoints": [N,K,3]) }
        """
        # Keep a local unique list of img_ids for this update; append to global buffer
        img_ids = list(np.unique(list(predictions.keys())))
        self.img_ids.extend(img_ids)

        for iou_type in self.iou_types:
            # Prepare results in the right COCO json-like format for the chosen iou_type
            results = self.prepare(predictions, iou_type)
            # Build detections COCO api. If no results, make an empty COCO to keep shapes valid.
            coco_dt = loadRes(self.coco_gt, results) if results else COCO()
            coco_eval = self.coco_eval[iou_type]

            # Assign detections & restrict evaluation to the updated image ids
            coco_eval.cocoDt = coco_dt
            coco_eval.params.imgIds = list(img_ids)
            # Run the per-image evaluation; returns (img_ids, evalImgs array)
            img_ids, eval_imgs = evaluate(coco_eval)

            # Stash the shard for later global merge/sync (e.g., across workers)
            self.eval_imgs[iou_type].append(eval_imgs)

    def synchronize_between_processes(self):
        """
        Merge shards (img_ids, evalImgs) produced by multiple update() calls,
        typically across distributed workers, and build a common COCOeval ready
        for `accumulate()`/`summarize()`.
        """
        for iou_type in self.iou_types:
            # Concatenate along the third axis: (cats, areas, imgs) -> imgs dimension grows
            self.eval_imgs[iou_type] = np.concatenate(self.eval_imgs[iou_type], 2)
            create_common_coco_eval(
                self.coco_eval[iou_type], self.img_ids, self.eval_imgs[iou_type]
            )

    def accumulate(self):
        """Run COCOeval.accumulate() for each IoU type to compute PR curves/stats."""
        for coco_eval in self.coco_eval.values():
            coco_eval.accumulate()

    def summarize(self):
        """Print the standard COCO summary table(s) for all configured IoU types."""
        for iou_type, coco_eval in self.coco_eval.items():
            print("IoU metric: {}".format(iou_type))
            coco_eval.summarize()

    def prepare(self, predictions, iou_type):
        """Dispatch formatter according to COCO IoU type."""
        if iou_type == "bbox":
            return self.prepare_for_coco_detection(predictions)
        elif iou_type == "segm":
            return self.prepare_for_coco_segmentation(predictions)
        elif iou_type == "keypoints":
            return self.prepare_for_coco_keypoint(predictions)
        else:
            raise ValueError("Unknown iou type {}".format(iou_type))

    def prepare_for_coco_detection(self, predictions):
        """
        Flatten a detection dict into a COCO results list of dicts with fields:
        image_id, category_id, bbox(x,y,w,h), score.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()  # COCO expects xywh
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)
                ]
            )
        return coco_results

    def prepare_for_coco_segmentation(self, predictions):
        """
        Convert binary masks to compressed RLE and build COCO-format results
        with image_id, category_id, segmentation (RLE), score.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            scores = prediction["scores"]
            labels = prediction["labels"]
            masks = prediction["masks"]

            masks = masks > 0.5  # binarize/threshold mask logits or probabilities

            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()

            # COCO requires Fortran-contiguous arrays for RLE encoding
            rles = [
                mask_util.encode(np.array(mask[0, :, :, np.newaxis], order="F"))[0]
                for mask in masks
            ]

            # pycocotools returns bytes for 'counts'; convert to utf-8 strings
            for rle in rles:
                rle["counts"] = rle["counts"].decode("utf-8")

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "segmentation": rle,
                        "score": scores[k],
                    }
                    for k, rle in enumerate(rles)
                ]
            )
        return coco_results

    def prepare_for_coco_keypoint(self, predictions):
        """
        Convert keypoints [N,K,3] to flat list [3K] per instance for COCO,
        alongside image_id, category_id, score.
        """
        coco_results = []
        for original_id, prediction in predictions.items():
            if len(prediction) == 0:
                continue

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            scores = prediction["scores"].tolist()
            labels = prediction["labels"].tolist()
            keypoints = prediction["keypoints"]
            keypoints = keypoints.flatten(start_dim=1).tolist()

            coco_results.extend(
                [
                    {
                        "image_id": original_id,
                        "category_id": labels[k],
                        "keypoints": keypoint,
                        "score": scores[k],
                    }
                    for k, keypoint in enumerate(keypoints)
                ]
            )
        return coco_results


def convert_to_xywh(boxes):
    """Convert xyxy boxes to xywh as required by COCO results json."""
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)


def merge(img_ids, eval_imgs):
    """
    Gather and merge shards across processes.

    Parameters
    ----------
    img_ids : list[int]
        Image ids collected locally.
    eval_imgs : np.ndarray
        Per-image eval arrays produced by pycocotools (cat x area x img dims).

    Returns
    -------
    merged_img_ids : np.ndarray
        Unique, sorted image ids.
    merged_eval_imgs : np.ndarray
        evalImgs aligned to the unique image ids.
    """
    # utils.all_gather performs distributed gather across ranks
    all_img_ids = utils.all_gather(img_ids)
    all_eval_imgs = utils.all_gather(eval_imgs)

    # Concatenate lists of ids
    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    # Stack evalImgs shards along the "imgs" dimension (axis=2 later)
    merged_eval_imgs = []
    for p in all_eval_imgs:
        merged_eval_imgs.append(p)

    merged_img_ids = np.array(merged_img_ids)
    merged_eval_imgs = np.concatenate(merged_eval_imgs, 2)

    # Keep unique (and ordered) image ids and align evalImgs accordingly
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_imgs = merged_eval_imgs[..., idx]

    return merged_img_ids, merged_eval_imgs


def create_common_coco_eval(coco_eval, img_ids, eval_imgs):
    """
    After merging, populate the shared COCOeval object fields so that
    `accumulate()`/`summarize()` operate on the full, merged dataset.
    """
    img_ids, eval_imgs = merge(img_ids, eval_imgs)
    img_ids = list(img_ids)
    eval_imgs = list(eval_imgs.flatten())

    coco_eval.evalImgs = eval_imgs
    coco_eval.params.imgIds = img_ids
    coco_eval._paramsEval = copy.deepcopy(coco_eval.params)


#################################################################
# From pycocotools, just removed the prints and fixed
# a Python3 bug about unicode not defined
#################################################################

# Ideally, pycocotools wouldn't have hard-coded prints
# so that we could avoid copy-pasting those two functions


def createIndex(self):
    """
    (Monkey-patched from pycocotools) Build fast lookup indices on the COCO api:
    - anns, imgs, cats: id -> object
    - imgToAnns: image_id -> [anns ...]
    - catToImgs: category_id -> [image_ids ...]
    """
    # create index
    # print('creating index...')
    anns, cats, imgs = {}, {}, {}
    imgToAnns, catToImgs = defaultdict(list), defaultdict(list)
    if "annotations" in self.dataset:
        for ann in self.dataset["annotations"]:
            imgToAnns[ann["image_id"]].append(ann)
            anns[ann["id"]] = ann

    if "images" in self.dataset:
        for img in self.dataset["images"]:
            imgs[img["id"]] = img

    if "categories" in self.dataset:
        for cat in self.dataset["categories"]:
            cats[cat["id"]] = cat

    if "annotations" in self.dataset and "categories" in self.dataset:
        for ann in self.dataset["annotations"]:
            catToImgs[ann["category_id"]].append(ann["image_id"])

    # print('index created!')

    # create class members
    self.anns = anns
    self.imgToAnns = imgToAnns
    self.catToImgs = catToImgs
    self.imgs = imgs
    self.cats = cats


maskUtils = mask_util


def loadRes(self, resFile):
    """
    (Monkey-patched from pycocotools) Load result json/list/ndarray and return a
    result COCO api object aligned with the GT api (images/categories copied).
    Accepts: path to json, numpy array, or pre-constructed list[dict].
    """
    res = COCO()
    res.dataset["images"] = [img for img in self.dataset["images"]]

    # print('Loading and preparing results...')
    # tic = time.time()
    if isinstance(resFile, _six.string_classes):
        anns = json.load(open(resFile))
    elif type(resFile) == np.ndarray:
        anns = self.loadNumpyAnnotations(resFile)
    else:
        anns = resFile

    assert type(anns) == list, "results in not an array of objects"
    annsImgIds = [ann["image_id"] for ann in anns]
    assert set(annsImgIds) == (
        set(annsImgIds) & set(self.getImgIds())
    ), "Results do not correspond to current coco set"

    if "caption" in anns[0]:
        imgIds = set([img["id"] for img in res.dataset["images"]]) & set(
            [ann["image_id"] for ann in anns]
        )
        res.dataset["images"] = [
            img for img in res.dataset["images"] if img["id"] in imgIds
        ]
        for id, ann in enumerate(anns):
            ann["id"] = id + 1
    elif "bbox" in anns[0] and not anns[0]["bbox"] == []:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            bb = ann["bbox"]
            x1, x2, y1, y2 = [bb[0], bb[0] + bb[2], bb[1], bb[1] + bb[3]]
            if "segmentation" not in ann:
                # Synthesize a polygon segmentation that matches the bbox (COCO-compatible)
                ann["segmentation"] = [[x1, y1, x1, y2, x2, y2, x2, y1]]
            ann["area"] = bb[2] * bb[3]
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "segmentation" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            # Only RLE is supported here; compute area and ensure bbox exists
            ann["area"] = maskUtils.area(ann["segmentation"])
            if "bbox" not in ann:
                ann["bbox"] = maskUtils.toBbox(ann["segmentation"])
            ann["id"] = id + 1
            ann["iscrowd"] = 0
    elif "keypoints" in anns[0]:
        res.dataset["categories"] = copy.deepcopy(self.dataset["categories"])
        for id, ann in enumerate(anns):
            s = ann["keypoints"]
            x = s[0::3]
            y = s[1::3]
            x0, x1, y0, y1 = np.min(x), np.max(x), np.min(y), np.max(y)
            ann["area"] = (x1 - x0) * (y1 - y0)
            ann["id"] = id + 1
            ann["bbox"] = [x0, y0, x1 - x0, y1 - y0]
    # print('DONE (t={:0.2f}s)'.format(time.time()- tic))

    res.dataset["annotations"] = anns
    createIndex(res)
    return res


def evaluate(self):
    """
    (Monkey-patched from pycocotools) Run per-image evaluation for the selected
    iouType over the configured image/category/area/det caps. Returns the image
    ids and a shaped numpy array of evalImgs for later accumulation.

    Notes
    -----
    This variant avoids noisy prints and returns `(imgIds, evalImgs)` so callers
    can aggregate across processes before running `accumulate()` / `summarize()`.
    """
    # tic = time.time()
    # print('Running per image evaluation...')
    p = self.params
    # add backward compatibility if useSegm is specified in params
    if p.useSegm is not None:
        p.iouType = "segm" if p.useSegm == 1 else "bbox"
        print(
            "useSegm (deprecated) is not None. Running {} evaluation".format(p.iouType)
        )
    # print('Evaluate annotation type *{}*'.format(p.iouType))
    p.imgIds = list(np.unique(p.imgIds))
    if p.useCats:
        p.catIds = list(np.unique(p.catIds))
    p.maxDets = sorted(p.maxDets)
    self.params = p

    self._prepare()
    # loop through images, area range, max detection number
    catIds = p.catIds if p.useCats else [-1]

    if p.iouType == "segm" or p.iouType == "bbox":
        computeIoU = self.computeIoU
    elif p.iouType == "keypoints":
        computeIoU = self.computeOks

    # Pre-compute IoUs/OKS for (img,cat) pairs used by evaluateImg
    self.ious = {
        (imgId, catId): computeIoU(imgId, catId)
        for imgId in p.imgIds
        for catId in catIds
    }

    evaluateImg = self.evaluateImg
    maxDet = p.maxDets[-1]

    # Evaluate each (img,cat,area) triple, restricted by maxDet
    evalImgs = [
        evaluateImg(imgId, catId, areaRng, maxDet)
        for catId in catIds
        for areaRng in p.areaRng
        for imgId in p.imgIds
    ]

    # Shape to (nCats, nAreaRngs, nImgs) for easy concat/merge later
    evalImgs = np.asarray(evalImgs).reshape(len(catIds), len(p.areaRng), len(p.imgIds))
    self._paramsEval = copy.deepcopy(self.params)
    # toc = time.time()
    # print('DONE (t={:0.2f}s).'.format(toc-tic))
    return p.imgIds, evalImgs


#################################################################
# end of straight copy from pycocotools, just removing the prints
#################################################################
