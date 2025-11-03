import os
import copy
import torch
from PIL import Image
import torch.utils.data
import torchvision
from pycocotools.coco import COCO
from pycocotools import mask as coco_mask

from . import transforms as T


class FilterAndRemapCocoCategories(object):
    """
    Filter annotations to a whitelist of category IDs and (optionally) remap
    the surviving category IDs to a compact 0..(N-1) range in-place.

    Parameters
    ----------
    categories : list[int]
        Whitelist of COCO category ids to keep.
    remap : bool
        If True, replace each kept `category_id` with its index in `categories`
        (i.e., contiguous labels starting at 0). If False, keep original ids.
    """

    def __init__(self, categories, remap=True):
        self.categories = categories
        self.remap = remap

    def __call__(self, image, target):
        # `target` is a raw COCO-dict with key 'annotations' (list of objs).
        anno = target["annotations"]
        # Keep only objects whose category_id is in the whitelist.
        anno = [obj for obj in anno if obj["category_id"] in self.categories]
        if not self.remap:
            # No remapping: just write filtered annotations and return.
            target["annotations"] = anno
            return image, target
        # With remapping: deep copy so we don't mutate the caller's objects.
        anno = copy.deepcopy(anno)
        for obj in anno:
            # Map original id -> index position in the whitelist for contiguous labels.
            obj["category_id"] = self.categories.index(obj["category_id"])
        target["annotations"] = anno
        return image, target


def convert_coco_poly_to_mask(segmentations, height, width):
    """
    Convert COCO polygon segmentations into a boolean mask tensor.

    Parameters
    ----------
    segmentations : list[list[list[float]]]
        COCO-style polygon lists per instance.
    height, width : int
        Target mask size.

    Returns
    -------
    masks : torch.Tensor [N, H, W] (dtype=uint8)
        Stacked instance masks (True/1 where inside the polygon).
        If no segmentations, returns an empty (0,H,W) tensor.
    """
    masks = []
    for polygons in segmentations:
        # Convert polygon lists to RLEs using pycocotools.
        rles = coco_mask.frPyObjects(polygons, height, width)
        # Decode RLEs to a (H, W, N) array (Fortran order expected by the API).
        mask = coco_mask.decode(rles)

        # Ensure we always have a third dimension for stacking later.
        if len(mask.shape) < 3:
            mask = mask[..., None]

        # Torch-ify and collapse along the instance axis (any polygon channel).
        mask = torch.as_tensor(mask, dtype=torch.uint8)
        mask = mask.any(dim=2)
        masks.append(mask)
    
    if masks:
        masks = torch.stack(masks, dim=0)
    else:
        masks = torch.zeros((0, height, width), dtype=torch.uint8)

    return masks


class ConvertCocoPolysToMask(object):
    """
    Transform a COCO-dataset sample (PIL image, raw COCO target) into
    a normalized detection target with tensors for boxes/labels/masks/etc.

    Notes
    -----
    - Filters out crowded annotations (`iscrowd == 0` kept).
    - Converts boxes from COCO xywh to xyxy and clamps to image size.
    - Optionally parses keypoints if present.
    - Adds fields needed for conversion back to COCO (area, iscrowd).
    """

    def __call__(self, image, target):
        # Original image size as (width, height)
        w, h = image.size

        # COCO image_id as a 1-D tensor; many training loops expect this shape.
        image_id = target["image_id"]
        image_id = torch.tensor([image_id])

        # Raw list of annotation dicts from COCO.
        anno = target["annotations"]

        # Exclude crowd regions for standard detection training.
        anno = [obj for obj in anno if obj["iscrowd"] == 0]

        # COCO bboxes are xywh; convert to a float32 tensor and then to xyxy.
        boxes = [obj["bbox"] for obj in anno]
        # guard against no boxes via resizing
        boxes = torch.as_tensor(boxes, dtype=torch.float32).reshape(-1, 4)
        boxes[:, 2:] += boxes[:, :2]
        # Clip coordinates to image bounds to avoid invalid boxes.
        boxes[:, 0::2].clamp_(min=0, max=w)
        boxes[:, 1::2].clamp_(min=0, max=h)

        # Class labels as int64 tensor.
        classes = [obj["category_id"] for obj in anno]
        classes = torch.tensor(classes, dtype=torch.int64)

        # Convert polygons to binary masks (N,H,W). Returns empty if none.
        segmentations = [obj["segmentation"] for obj in anno]
        masks = convert_coco_poly_to_mask(segmentations, h, w)

        # Optional keypoints handling (only if any annotation contains the field).
        keypoints = None    
        if anno and "keypoints" in anno[0]:
            keypoints = [obj["keypoints"] for obj in anno]
            keypoints = torch.as_tensor(keypoints, dtype=torch.float32)
            num_keypoints = keypoints.shape[0]
            if num_keypoints:
                # Reshape to [N, K, 3] where (x,y,visibility) triplets are grouped.
                keypoints = keypoints.view(num_keypoints, -1, 3)

        # Drop invalid/degenerate boxes (x2>x1 and y2>y1).
        keep = (boxes[:, 3] > boxes[:, 1]) & (boxes[:, 2] > boxes[:, 0])
        boxes = boxes[keep]
        classes = classes[keep]
        masks = masks[keep]
        if keypoints is not None:
            keypoints = keypoints[keep]

        # Build the normalized training target dict.
        target = {}
        target["boxes"] = boxes
        target["labels"] = classes
        target["masks"] = masks
        target["image_id"] = image_id
        if keypoints is not None:
            target["keypoints"] = keypoints

        # Keep COCO-specific fields for potential round-trips/back-exports.
        area = torch.tensor([obj["area"] for obj in anno])
        iscrowd = torch.tensor([obj["iscrowd"] for obj in anno])
        target["area"] = area
        target["iscrowd"] = iscrowd

        return image, target


def _coco_remove_images_without_annotations(dataset, cat_list=None):
    """
    Filter out images that don't have valid annotations for training.

    Criteria
    --------
    - No annotations at all -> drop.
    - All boxes effectively empty (w<=1 or h<=1) -> drop.
    - For keypoints tasks: require at least `min_keypoints_per_image` visible points.

    Parameters
    ----------
    dataset : torchvision.datasets.CocoDetection
        A standard COCO detection dataset instance.
    cat_list : Optional[list[int]]
        If provided, consider only annotations with category_id in this list.

    Returns
    -------
    torch.utils.data.Subset
        A subset view containing only valid image indices.
    """

    def _has_only_empty_bbox(anno):
        # True if every bbox width/height is <= 1 pixel.
        return all(any(o <= 1 for o in obj["bbox"][2:]) for obj in anno)

    def _count_visible_keypoints(anno):
        # Count #visible keypoints (v>0) across all annotations.
        return sum(sum(1 for v in ann["keypoints"][2::3] if v > 0) for ann in anno)

    min_keypoints_per_image = 10

    def _has_valid_annotation(anno):
        # if it's empty, there is no annotation
        if len(anno) == 0:
            return False
        
        # if all boxes have close to zero area, there is no annotation
        if _has_only_empty_bbox(anno):
            return False
        
        # keypoints task have a slight different critera for considering
        # if an annotation is valid
        if "keypoints" not in anno[0]:
            return True
        
        # for keypoint detection tasks, only consider valid images those
        # containing at least min_keypoints_per_image
        if _count_visible_keypoints(anno) >= min_keypoints_per_image:
            return True
        return False

    assert isinstance(dataset, torchvision.datasets.CocoDetection)
    ids = []
    for ds_idx, img_id in enumerate(dataset.ids):
        # Fetch this image's annotation list from the underlying COCO api.
        ann_ids = dataset.coco.getAnnIds(imgIds=img_id, iscrowd=None)
        anno = dataset.coco.loadAnns(ann_ids)

        # If category filter is provided, apply it before validation.
        if cat_list:
            anno = [obj for obj in anno if obj["category_id"] in cat_list]

        if _has_valid_annotation(anno):
            ids.append(ds_idx)

    # Wrap the dataset with a Subset adapter to keep only valid indices.
    dataset = torch.utils.data.Subset(dataset, ids)
    return dataset


def convert_to_coco_api(ds):
    """
    Convert an arbitrary detection dataset (following torchvision targets) into
    a COCO API object (COCO()) by synthesizing the in-memory COCO json dict.

    Expectations for `ds`
    ---------------------
    - `ds[i]` returns (image_tensor[C,H,W], target_dict).
    - target_dict has keys: image_id(int), boxes[t,4] (xyxy), labels[t],
      area[t], iscrowd[t], optionally masks[t,H,W] and keypoints[t,K,3].

    Returns
    -------
    coco_ds : COCO
        COCO object with images/categories/annotations populated, and indexed.
    """
    coco_ds = COCO()
    ann_id = 0
    dataset = {"images": [], "categories": [], "annotations": []}
    categories = set()
    for img_idx in range(len(ds)):
        # NOTE: If the dataset exposes a different API, adapt here accordingly.
        # find better way to get target
        # targets = ds.get_annotations(img_idx)
        img, targets = ds[img_idx]
        image_id = targets["image_id"].item()
        img_dict = {}
        img_dict["id"] = image_id
        # Expect image tensors (C,H,W) here; use last two dims.
        img_dict["height"] = img.shape[-2]
        img_dict["width"] = img.shape[-1]
        dataset["images"].append(img_dict)
        # Convert predicted xyxy -> xywh for COCO storage.
        bboxes = targets["boxes"]
        bboxes[:, 2:] -= bboxes[:, :2]
        bboxes = bboxes.tolist()
        labels = targets["labels"].tolist()
        areas = targets["area"].tolist()
        iscrowd = targets["iscrowd"].tolist()
        if "masks" in targets:
            masks = targets["masks"]
            # Ensure Fortran-contiguous layout for pycocotools encoding.
            masks = masks.permute(0, 2, 1).contiguous().permute(0, 2, 1)
        if "keypoints" in targets:
            keypoints = targets["keypoints"]
            keypoints = keypoints.reshape(keypoints.shape[0], -1).tolist()
        num_objs = len(bboxes)
        for i in range(num_objs):
            ann = {}
            ann["image_id"] = image_id
            ann["bbox"] = bboxes[i]
            ann["category_id"] = labels[i]
            categories.add(labels[i])
            ann["area"] = areas[i]
            ann["iscrowd"] = iscrowd[i]
            ann["id"] = ann_id

            if "masks" in targets:
                # RLE encoding of the i-th mask
                ann["segmentation"] = coco_mask.encode(masks[i].numpy())

            if "keypoints" in targets:
                ann["keypoints"] = keypoints[i]
                # Number of visible keypoints (#(v != 0))
                ann["num_keypoints"] = sum(k != 0 for k in keypoints[i][2::3])

            dataset["annotations"].append(ann)
            ann_id += 1
    
    # COCO requires a `categories` list with dicts at least containing id's.
    dataset["categories"] = [{"id": i} for i in sorted(categories)]
    coco_ds.dataset = dataset
    coco_ds.createIndex()
    return coco_ds


def get_coco_api_from_dataset(dataset):
    """
    Retrieve a COCO api (COCO) for a given dataset. If the dataset is already a
    CocoDetection, return its `.coco`. If it's a Subset, unwrap it; otherwise
    synthesize a COCO api using `convert_to_coco_api`.

    Parameters
    ----------
    dataset : Dataset | Subset
        Any dataset following torchvision conventions.

    Returns
    -------
    COCO
        A COCO api instance describing `dataset`.
    """

    for i in range(10):
        if isinstance(dataset, torchvision.datasets.CocoDetection):
            break
        if isinstance(dataset, torch.utils.data.Subset):
            dataset = dataset.dataset
    
    if isinstance(dataset, torchvision.datasets.CocoDetection):
        return dataset.coco
    
    return convert_to_coco_api(dataset)


class CocoDetection(torchvision.datasets.CocoDetection):
    """
    Thin subclass of torchvision's CocoDetection that injects our transform
    pipeline and wraps the raw annotations into the expected training target
    format used by the rest of this codebase.
    """

    def __init__(self, img_folder, ann_file, transforms):
        super(CocoDetection, self).__init__(img_folder, ann_file)
        self._transforms = transforms

    def __getitem__(self, idx):
        # Get PIL image + raw list of COCO annotation dicts from parent.
        img, target = super(CocoDetection, self).__getitem__(idx)
        image_id = self.ids[idx]
        # Wrap into the dict format expected by downstream transforms/loops.
        target = dict(image_id=image_id, annotations=target)

        # Apply composed transforms (includes ConvertCocoPolysToMask + user transforms).
        if self._transforms is not None:
            img, target = self._transforms(img, target)
        return img, target


def get_coco(root, image_set, transforms, mode="instances"):
    """
    Build a standard COCO dataset (train/val) with conversion/transforms applied.

    Parameters
    ----------
    root : str
        Root directory containing COCO folders (train2017/val2017 and annotations/).
    image_set : {"train","val"}
        Which split to load.
    transforms : callable or None
        User-provided transform (will be composed after ConvertCocoPolysToMask).
    mode : {"instances","person_keypoints"}
        Annotation mode to select.

    Returns
    -------
    CocoDetection | Subset
        A dataset ready for training/evaluation; for "train" split, images
        without valid annotations are removed.
    """

    anno_file_template = "{}_{}2017.json"
    PATHS = {
        "train": (
            "train2017",
            os.path.join("annotations", anno_file_template.format(mode, "train")),
        ),
        "val": (
            "val2017",
            os.path.join("annotations", anno_file_template.format(mode, "val")),
        ),
        # "train": ("val2017", os.path.join("annotations", anno_file_template.format(mode, "val")))
    }

    # Always convert polygons -> masks first; then apply user transforms (if any).
    t = [ConvertCocoPolysToMask()]

    if transforms is not None:
        t.append(transforms)
    transforms = T.Compose(t)

    img_folder, ann_file = PATHS[image_set]
    img_folder = os.path.join(root, img_folder)
    ann_file = os.path.join(root, ann_file)

    dataset = CocoDetection(img_folder, ann_file, transforms=transforms)

    if image_set == "train":
        # Remove images that don't meet training criteria (no/invalid annos).
        dataset = _coco_remove_images_without_annotations(dataset)

    # You can optionally subset for quick experiments (see commented line below).
    # dataset = torch.utils.data.Subset(dataset, [i for i in range(500)])

    return dataset


def get_coco_kp(root, image_set, transforms):
    """
    Convenience wrapper for the COCO keypoints task. Equivalent to
    `get_coco(..., mode="person_keypoints")`.
    """
    return get_coco(root, image_set, transforms, mode="person_keypoints")
