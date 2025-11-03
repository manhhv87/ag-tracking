import cv2
import torch
import numpy as np
from collections import deque
import torch.nn.functional as F
from torchvision.ops import box_iou
from scipy.optimize import linear_sum_assignment
from torchvision.ops.boxes import clip_boxes_to_image, nms
from .utils import warp_pos, get_center, get_height, get_width, make_pos


class Tracker:
    """
    Multi-object tracker that follows a Tracktor-style tracking-by-detection paradigm
    adapted to agricultural scenes.

    High-level behavior (informed by FloraTracktor / FloraTracktor+):
      • Maintain a set of active tracks and an inactive pool.
      • At each frame: (1) regress active tracks with a detector head; (2) fuse
        regressed tracks with raw detections via a *single* NMS pass to resolve
        conflicts/overlaps; (3) keep tracks above a 'keep-alive' threshold and
        create new tracks from detections above a 'new' threshold.
      • Store per-track history and export frame-wise results.

    Notes vs. the paper:
      • The paper augments Tracktor with a Spatial Association Module (LoFTR-based)
        to estimate global motion and strengthen re-association. This implementation
        focuses on the detector+regression+NMS portion that the paper also highlights
        (Fig. 1 pipeline) and uses confidence thresholds analogous to s_active,
        s_new, and λ_nms studied in the sensitivity analysis (Fig. 2).
    """

    def __init__(self, detector, tracker_cfg):
        # Detector providing predict_boxes(); thresholds and NMS from config.
        self.detector = detector        
        self.detection_person_thresh = tracker_cfg.DETECTION_PERSON_THRESH      # ≈ s_new
        self.regression_person_thresh = tracker_cfg.REGRESSION_PERSON_THRESH    # ≈ s_active
        self.nms_thresh = tracker_cfg.NMS_THRESH    # λ_nms

        # Active and inactive track pools; monotonically increasing id counter; frame idx.
        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0

        # results[track_id][frame_idx] = [x1,y1,x2,y2,score]
        self.results = {}

    def reset(self, hard=True):
        """
        Reset tracker state.

        Args:
            hard (bool): If True, also reset counters and stored results.
        """
        self.tracks = []
        self.inactive_tracks = []
        if hard:
            self.track_num = 0
            self.im_index = 0
            self.results = {}

    def tracks_to_inactive(self, tracks):
        """
        Move a batch of tracks from active to inactive, freezing their current position.

        Rationale: mirrors the paper's idea of temporarily deactivating tracks that
        cannot be confidently kept alive, while preserving state for possible future
        use (e.g., re-entry logic).
        """
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]  # keep last confident box
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores):
        """
        Initialize new Track objects from detection boxes that survived NMS
        and exceed the 'new track' confidence threshold.

        Args:
            new_det_pos (Tensor[N,4]): xyxy boxes in current frame.
            new_det_scores (Tensor[N]): detector confidences.
        """
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            # Give each new track a unique id; keep a copy of its initial box.
            self.tracks.append(
                Track(new_det_pos[i].view(1, -1), new_det_scores[i], self.track_num + i)
            )
        self.track_num += num_new

    def regress_tracks(self, blob):
        """
        Regress active tracks' boxes on the current frame using the detector head,
        then clamp boxes to the image bounds and deactivate tracks with zero scores.

        Args:
            blob (dict): expects
                - "features": backbone/ROI features for the current frame
                - "img": the input image tensor (for spatial size)
        """
        if len(self.tracks):
            # Gather current positions to query the detector head.
            pos, _ = self.get_pos()

            # Predict updated boxes and scores for each active track (class-agnostic).
            boxes, scores = self.detector.predict_boxes(
                pos, blob["features"], 1
            )  # class id

            # Keep boxes within image; prevents negative/overflow coords.
            pos = clip_boxes_to_image(boxes, blob["img"].shape[-2:])

            # Update tracks in-place (reverse order to allow safe removal).
            for i in range(len(self.tracks) - 1, -1, -1):
                t = self.tracks[i]
                t.score = scores[i]
                t.pos = pos[i].view(1, -1)

                # Deactivate tracks that become invalid (e.g., regression failure).
                if scores[i] == 0:
                    self.tracks_to_inactive([t])

    def get_pos(self):
        """
        Returns:
            pos (Tensor[M,4]): stacked xyxy boxes of active tracks (M may be 0).
            score (Tensor[M]): corresponding confidences (CUDA tensors for ops).
        """
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
            score = torch.Tensor([self.tracks[0].score.item()]).cuda()
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], 0)
            score = torch.Tensor([t.score.item() for t in self.tracks]).cuda()
        else:
            pos = torch.zeros(0).cuda()
            score = torch.zeros(0).cuda()
        return pos, score

    def step(self, blob, detections):
        """
        Advance the tracker by one frame.

        This performs the sequence:
          1) Save current positions to per-track history.
          2) Filter raw detections by class id.
          3) Regress active tracks.
          4) Run a **single NMS over detections + regressed tracks** to:
             • keep high-confidence regressed tracks (keep-alive),
             • replace a track with a better detection if it suppresses the track,
             • create new tracks from surviving detections above s_new,
             • deactivate low-confidence tracks.
          5) Write results and maintain the inactive pool.

        Args:
            blob (dict): frame image/features; see `regress_tracks`.
            detections (dict): keys = "boxes", "scores", "labels".
        """
        for t in self.tracks:
            # add current position to last_pos list. BH 1
            # Maintain a short history window; used to prefer longer-lived tracks
            # if multiple tracks are suppressed by the same detection.
            t.last_pos.append(t.pos.clone())

        # filter by class id
        boxes, scores, labels = (
            detections["boxes"],
            detections["scores"],
            detections["labels"],
        )
        det_pos = boxes[labels == 1]  # class id
        det_scores = scores[
            labels == 1
        ]  # class id

        # Predict tracks
        if len(self.tracks):
            # 1) Update regressed boxes for current frame.
            self.regress_tracks(blob)

        # BH: Combined approach to filter using nms
        if det_pos.nelement() > 0 and len(self.tracks) > 0:
            # Build the joint set: [regressed tracks; raw detections].
            pos, score = self.get_pos()
            poses = torch.cat([pos, det_pos])
            scores = torch.cat([score, det_scores])
            # Indices and types help map NMS decisions back to tracks/dets.
            # d: new detections, t: existing tracks
            pos_idxs = list(range(len(self.tracks))) + list(range(det_pos.shape[0]))
            pos_type = len(self.tracks) * ["t"] + det_pos.shape[0] * ["d"]
            ious = box_iou(poses, poses)
            nms_keep = nms(poses, scores, self.nms_thresh)
            det_pos_keep = torch.zeros(det_pos.shape[0]).cuda().bool()
            trk_pos_keep = torch.zeros(len(self.tracks)).cuda().bool()
            for idx in nms_keep:
                if pos_type[idx] == "d":  # detection
                    replaced = False  # replaced a track
                    # Find any tracks suppressed by this kept detection.
                    supress = torch.gt(ious[idx], self.nms_thresh)
                    indexes, lens = [], []
                    for i, sup in enumerate(supress):
                        if (
                            sup.item() and i != idx.item() and pos_type[i] == "t"
                        ):  # if the detection supressed a track
                            replaced = True
                            # Replace the suppressed track's box/score with the detection.
                            self.tracks[pos_idxs[i]].pos = poses[idx].view(1, -1)
                            self.tracks[pos_idxs[i]].score = scores[idx]
                            indexes.append(pos_idxs[i])
                            # Track history length used to prefer a single survivor later.
                            lens.append(len(self.tracks[indexes[-1]].last_pos))
                    
                    if replaced:
                        # Keep only the longest-lived of the replaced tracks.
                        trk_pos_keep[indexes[lens.index(max(lens))]] = True

                    if (
                        not replaced
                        and scores[idx].item() > self.detection_person_thresh
                    ):
                        # No track was suppressed -> this detection can start a new track.
                        det_pos_keep[pos_idxs[idx]] = True
                else:  # existing track
                    # Surviving regressed tracks must clear the keep-alive threshold.
                    if scores[idx].item() > self.regression_person_thresh:
                        trk_pos_keep[pos_idxs[idx]] = True

            # Filter detections to become new tracks; deactivate tracks not kept.
            det_pos = det_pos[det_pos_keep]
            det_scores = det_scores[det_pos_keep]
            self.tracks_to_inactive(
                [t for i, t in enumerate(self.tracks) if not trk_pos_keep[i].item()]
            )

        elif det_pos.nelement() > 0:
            # No active tracks: standard NMS on raw detections, then create tracks.
            # BH: filtering by NMS between raw detections to filter
            keep = nms(det_pos, det_scores, self.nms_thresh)
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

        # Create new tracks
        if det_pos.nelement() > 0:
            self.add(det_pos, det_scores)

        # Generate Results
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}

            # Save [x1,y1,x2,y2,score] for this frame and track id.
            self.results[t.id][self.im_index] = np.concatenate(
                [t.pos[0].cpu(), np.array([t.score.cpu()])]
            )

        # Age inactive tracks (for optional time-based pruning downstream).
        for t in self.inactive_tracks:
            t.count_inactive += 1

        # Drop obviously invalid boxes (negative/too small areas).
        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area()
        ]
        self.im_index += 1

    def get_results(self):
        """
        Returns:
            dict: results[track_id][frame_idx] -> np.ndarray(5,)
                  with [x1,y1,x2,y2,score] per frame.
        """
        return self.results


class Track(object):
    """
    Lightweight container for a single tracked instance.

    Attributes:
        id (int): unique identifier.
        pos (Tensor[1,4]): current xyxy box.
        score (Tensor|float): current confidence score.
        count_inactive (int): number of consecutive frames the track has been inactive.
        last_pos (deque): short history of recent boxes for tie-breaking and stability.
    """

    def __init__(self, pos, score, track_id, back=5):
        self.id = track_id
        self.pos = pos
        self.score = score
        self.count_inactive = 0
        # Keep a small window of previous boxes to favor longer-lived tracks
        # when a single detection suppresses multiple tracks.
        self.last_pos = deque([pos.clone()], maxlen=back)  # TODO: determine this

    def has_positive_area(self):
        """
        Basic sanity check to reject degenerate boxes (too small/non-positive area).
        The `target` margin avoids keeping tiny noisy boxes that could pollute results.
        """
        # is x2 > x1 and y2 > y1
        target = 15
        return (
            self.pos[0, 2] > self.pos[0, 0] + target
            and self.pos[0, 3] > self.pos[0, 1] + target
        )

    def reset_last_pos(self):
        """Clear history and re-seed with the current position."""
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())
