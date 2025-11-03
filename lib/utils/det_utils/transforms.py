import random
from torchvision.transforms import functional as F


def _flip_coco_person_keypoints(kps, width):
    """
    Flip COCO person keypoints horizontally.

    Parameters
    ----------
    kps : torch.Tensor [N, K, 3]
        Keypoints in COCO format per instance: (x, y, v) where v ∈ {0,1,2}.
        K must follow the COCO person keypoint order.
    width : int
        Image width in pixels (needed to mirror x-coordinates).

    Returns
    -------
    torch.Tensor [N, K, 3]
        Flipped keypoints with left/right parts swapped and x mirrored.

    Notes
    -----
    - `flip_inds` maps each COCO keypoint index to its left/right counterpart
      (e.g., left_eye ↔ right_eye).
    - After horizontal flip: x' = width - x.
    - COCO convention: if visibility v == 0 (not labeled), then (x, y) must be 0.
      This is preserved after flipping.
    """
    
    flip_inds = [0, 2, 1, 4, 3, 6, 5, 8, 7, 10, 9, 12, 11, 14, 13, 16, 15]
    flipped_data = kps[:, flip_inds]                        # swap left/right joints by index
    flipped_data[..., 0] = width - flipped_data[..., 0]     # mirror x about image width
    # Maintain COCO rule: when visibility==0, zero-out x,y
    inds = flipped_data[..., 2] == 0
    flipped_data[inds] = 0
    return flipped_data


class Compose(object):
    """Compose a list of (image, target) transforms into a single callable."""

    def __init__(self, transforms):
        """
        Parameters
        ----------
        transforms : list[callable]
            Each callable takes (image, target) and returns (image, target).
        """
        self.transforms = transforms

    def __call__(self, image, target):
        """
        Apply each transform in sequence.

        Returns
        -------
        (image, target) after all transforms have been applied.
        """
        for t in self.transforms:
            image, target = t(image, target)
        return image, target


class RandomHorizontalFlip(object):
    """
    Randomly flip image and annotations horizontally with probability `prob`.

    This transform:
    - Flips the image tensor left-right.
    - Updates bounding boxes: x1/x2 columns swapped and mirrored.
    - Flips binary masks if present.
    - Flips COCO keypoints (swap left/right indices and mirror x) if present.
    """

    def __init__(self, prob):
        """
        Parameters
        ----------
        prob : float in [0,1]
            Probability of applying the horizontal flip.
        """
        self.prob = prob

    def __call__(self, image, target):
        """
        Apply the flip consistently to image and target components.

        Parameters
        ----------
        image : torch.Tensor [C, H, W]
            Image tensor.
        target : dict
            Must contain:
              - "boxes": torch.Tensor [N,4] (x1,y1,x2,y2).
            May contain:
              - "masks": torch.Tensor [N,H,W] (uint8/bool).
              - "keypoints": torch.Tensor [N,K,3] (COCO format).

        Returns
        -------
        (image, target) : tuple
            Flipped or original (image, target) depending on random draw.
        """
        if random.random() < self.prob:
            # Get spatial size (H, W) from the tensor, flip along width dimension.
            height, width = image.shape[-2:]
            image = image.flip(-1)

            # Mirror bounding boxes: swap x1/x2 and mirror around image width.
            bbox = target["boxes"]
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # [x1,x2] := [W-x2, W-x1]
            target["boxes"] = bbox

            # If masks exist, flip them along width as well.
            if "masks" in target:
                target["masks"] = target["masks"].flip(-1)

            # If COCO keypoints exist, handle left/right index swapping and x mirror.
            if "keypoints" in target:
                keypoints = target["keypoints"]
                keypoints = _flip_coco_person_keypoints(keypoints, width)
                target["keypoints"] = keypoints
        return image, target


class ToTensor(object):
    """
    Convert a PIL Image or numpy array to a torch.FloatTensor in [0,1],
    leaving the `target` dictionary untouched.
    """

    def __call__(self, image, target):
        """
        Parameters
        ----------
        image : PIL.Image | np.ndarray
            Input image to convert.
        target : dict
            Passed through unchanged.

        Returns
        -------
        (image_tensor, target)
            `image_tensor` is torch.FloatTensor [C,H,W] scaled to [0,1].
        """
        image = F.to_tensor(image)
        return image, target
