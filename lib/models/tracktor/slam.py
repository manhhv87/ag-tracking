import cv2
import numpy as np
import torch
import kornia
import kornia.feature as KF


class SLAM:
    """
    Minimal SLAM-style tracker using LoFTR correspondences and Euclidean motion.

    Parameters
    ----------
    cfg : dict
        Must include:
        - "device": e.g., "cuda:0" or "cpu"
        - "loftr_cfg": configuration dict for LoFTR (passed to KF.LoFTR)
    factor : int
        Downsampling factor (images are resized by (H//factor, W//factor) for matching).
    ransac_t : float
        RANSAC reprojection threshold (in pixels) for affine estimation.

    Attributes
    ----------
    device : torch.device
        Device on which LoFTR and tensors will be placed.
    matcher : KF.LoFTR
        Pretrained LoFTR matcher in eval mode.
    factor : int
        Spatial reduction factor used before matching.
    ransac : float
        RANSAC reprojection error threshold.
    imgTpre : torch.Tensor or None
        Previous grayscale, downsampled tensor used as LoFTR's `image0`.
    H : np.ndarray, shape (3,3)
        Running homography-like transform (Euclidean 2D transform embedded in 3x3).
    """

    def __init__(self, cfg, factor, ransac_t):
        # Select compute device from config (e.g., "cuda:0")
        self.device = torch.device(cfg["device"])

        # Build LoFTR matcher with "outdoor" weights and given config; keep in eval mode
        self.matcher = KF.LoFTR(pretrained="outdoor", config=cfg["loftr_cfg"]).to(
            self.device
        )
        self.matcher.eval()

        # Store parameters controlling image resizing and RANSAC threshold
        self.factor = factor
        self.ransac = ransac_t

        # Previous downsampled grayscale image tensor (None until first frame processed)
        self.imgTpre = None

        # Initialize cumulative transform as identity (maps first frame to itself)
        self.H = np.eye(3)

    def run_frame(self, img, h, w, t, out=torch.tensor(0).cuda()):
        """
        Process one frame and update the cumulative transform `H`.

        Parameters
        ----------
        img : torch.Tensor
            Current frame as a **RGB** tensor (NCHW or CHW compatible with Kornia ops).
            This function expects a tensor on a device compatible with `self.device`.
        h, w : int
            Spatial size used for downsampling target: output will be (h//factor, w//factor).
        t : int
            Time/frame index: if `t == 0` (first frame) no matching or estimation is done.
        out : torch.Tensor
            Unused default argument kept exactly as in the original code.

        Steps
        -----
        1) Convert RGB tensor to grayscale on the target device.
        2) Resize to the reduced resolution for faster, more stable matching.
        3) If not the first frame:
           - run LoFTR on previous vs current reduced grayscale tensors,
           - rescale keypoints back by `factor` to original resolution,
           - estimate a 2D **Euclidean** transform with RANSAC,
           - normalize the 2x2 submatrix to unit scale,
           - compose it into the running transform `self.H`.
        4) Cache the current reduced tensor as `self.imgTpre`.
        """
        # Convert to grayscale (expects channel-first) and move to the configured device
        imgT = kornia.color.rgb_to_grayscale(img.to(self.device))

        # Downsample to speed up matching and improve robustness
        imgT = kornia.geometry.transform.resize(
            imgT, [h // self.factor, w // self.factor]
        )

        # Skip matching for the very first frame `t == 0`
        if t:
            # Prepare LoFTR input dict; previous frame is `image0`, current is `image1`
            imgdict = {"image0": self.imgTpre, "image1": imgT}

            # Run the matcher to obtain sparse correspondences between the two frames
            correspondences = self.matcher(imgdict)
            # print('Frame %d, Correspondences:' % i, len(correspondences["keypoints0"]))

            # Extract matched keypoints and rescale them back to the original resolution
            mkpts0 = correspondences["keypoints0"].cpu().numpy() * np.array(
                [self.factor, self.factor]
            )
            mkpts1 = correspondences["keypoints1"].cpu().numpy() * np.array(
                [self.factor, self.factor]
            )

            # Estimate a **partial affine** (rotation + translation + uniform scale)
            # We then normalize the scale to 1.0 to make it a pure Euclidean transform.
            H = np.eye(3)
            H[:2, :], _ = cv2.estimateAffinePartial2D(
                mkpts1, mkpts0, method=cv2.RANSAC, ransacReprojThreshold=self.ransac
            )

            # Remove scale: divide 2x2 by sqrt(det(R)) to keep rotation orthonormal
            H[:2, :2] = H[:2, :2] / np.sqrt(np.linalg.det(H[:2, :2]))

            # Compose the transform so that `self.H` maps from the first frame to the current frame
            self.H = self.H @ H

        # Cache the current downsampled grayscale tensor for the next iteration
        self.imgTpre = imgT.detach()


import torch.multiprocessing as mp


class SLAMProcess(mp.Process):
    """
    A `multiprocessing.Process` wrapper running the same LoFTR-based pipeline.

    Parameters
    ----------
    cfg : dict
        Same as `SLAM`: contains "device" and "loftr_cfg".
    factor : int
        Downsampling factor (H//factor, W//factor).
    ransac_t : float
        RANSAC threshold for `cv2.estimateAffinePartial2D`.

    Attributes
    ----------
    device : torch.device
        Compute device for the LoFTR matcher.
    matcher : KF.LoFTR
        LoFTR instance in eval mode.
    factor : int
        Downsampling factor.
    ransac : float
        RANSAC reprojection threshold.
    imgTpre : torch.Tensor or None
        Cached previous reduced grayscale tensor.
    args : tuple or None
        If not None, arguments used by `run_frame` when `run()` is invoked.
    H : np.ndarray, shape (3,3)
        Cumulative homography-like transform.
    """

    def __init__(self, cfg, factor, ransac_t):
        super(SLAMProcess, self).__init__()

        # Device and matcher are created in this separate process instance
        self.device = torch.device(cfg["device"])
        self.matcher = KF.LoFTR(pretrained="outdoor", config=cfg["loftr_cfg"]).to(
            self.device
        )
        self.matcher.eval()

        self.factor = factor
        self.ransac = ransac_t
        self.imgTpre = None

        # Optional deferred-args mechanism: if provided before `start()`,
        # `run()` will call `run_frame(*self.args)`.
        self.args = None

        # Start with identity transform
        self.H = np.eye(3)

    def run(self):
        """
        Entry point of the spawned process.

        Behavior
        --------
        - If `self.args` is set, call `self.run_frame(*self.args)`.
        - Otherwise, just print a small notice (kept as in original code).
        """
        if self.args:
            self.run_frame(*self.args)
        else:
            print("started the SLAM process...")

    def run_frame(self, img, h, w, t):
        """
        Same frame-processing logic as in `SLAM.run_frame`, without the extra default arg.

        Parameters
        ----------
        img : torch.Tensor
            RGB tensor for the current frame.
        h, w : int
            Height/width used to compute the reduced resolution (h//factor, w//factor).
        t : int
            Frame index; if zero, the method only initializes the cache.

        Notes
        -----
        - This implementation does not take a device argument; it expects that
          tensors are already on a device compatible with `self.device`.
        """
        # Convert to grayscale (channel-first) for LoFTR
        imgT = kornia.color.rgb_to_grayscale(img)

        # Downsample both spatial dimensions
        imgT = kornia.geometry.transform.resize(
            imgT, [h // self.factor, w // self.factor]
        )

        if t:
            # Prepare and run the LoFTR matcher on prev/current reduced frames
            imgdict = {"image0": self.imgTpre, "image1": imgT}
            correspondences = self.matcher(imgdict)
            # print('Frame %d, Correspondences:' % i, len(correspondences["keypoints0"]))

            # Bring keypoints back to original resolution by multiplying by factor
            mkpts0 = correspondences["keypoints0"].cpu().numpy() * np.array(
                [self.factor, self.factor]
            )
            mkpts1 = correspondences["keypoints1"].cpu().numpy() * np.array(
                [self.factor, self.factor]
            )

            # Estimate Euclidean (scale-normalized) 2D transform via partial affine
            H = np.eye(3)
            H[:2, :], _ = cv2.estimateAffinePartial2D(
                mkpts1, mkpts0, method=cv2.RANSAC, ransacReprojThreshold=self.ransac
            )
            H[:2, :2] = H[:2, :2] / np.sqrt(np.linalg.det(H[:2, :2]))

            # Accumulate into the running transform
            self.H = self.H @ H

        # Cache the downsampled grayscale tensor for the next frame
        self.imgTpre = imgT.detach()
