from collections import deque

import cv2
import numpy as np
import torch
from scipy.optimize import linear_sum_assignment

# from torchreid import metrics
import torch.nn.functional as F
from torchvision.ops.boxes import clip_boxes_to_image, nms

from .utils import bbox_overlaps, get_center, get_height, get_width, make_pos, warp_pos


class Tracker:
    """
    Multi-object tracker following a Tracktor-style tracking-by-detection paradigm,
    augmented with two stabilizers that echo ideas from FloraTracktor(+):
      1) Global image alignment (ECC homography-like warp) to compensate camera motion.
      2) Lightweight linear motion model using recent positions.

    Pipeline per frame:
      • (Optional) ECC alignment: warp previous boxes to current frame coordinates.
      • (Optional) Linear motion step: extrapolate boxes using average recent velocity.
      • Regress boxes of active tracks using the detector head; drop low-confidence tracks.
      • NMS among regressed tracks (keep only the best non-overlapping tracks).
      • Gate raw detections by score + NMS, then suppress detections already covered by tracks
        (by giving tracks a >1 score "priority" during NMS).
      • Re-identify (Hungarian) inactive tracks with remaining detections using appearance
        distance (and optional IoU gating); re-activate on successful matches.
      • Create *new* tracks from remaining high-confidence detections.
      • Record per-id results and age inactive tracks; prune hopeless ones.

    Notes vs. the paper:
      • The paper's Spatial Association Module (LoFTR-based) estimates homography and matches
        across frames without relying on appearance embeddings. Here, ECC-based alignment and
        an appearance ReID module are used to serve similar re-association/stabilization roles.
    """

    # only track pedestrian
    cl = 1

    def __init__(self, obj_detect, reid_network, tracker_cfg):
        # Detector (regression head) and ReID network (appearance features)
        self.obj_detect = obj_detect
        self.reid_network = reid_network

        # Confidence/NMS thresholds for (a) new detections and (b) regressed tracks.
        # Analogous to s_new, s_active, and λ_nms in the paper's sensitivity analysis.
        self.detection_person_thresh = tracker_cfg.DETECTION_PERSON_THRESH
        self.regression_person_thresh = tracker_cfg.REGRESSION_PERSON_THRESH
        self.detection_nms_thresh = tracker_cfg.DETECTION_NMS_THRESH
        self.regression_nms_thresh = tracker_cfg.REGRESSION_NMS_THRESH

        # Lifecycle & feature-queue parameters
        self.inactive_patience = tracker_cfg.INACTIVE_PATIENCE
        self.public_detections = False
        self.do_reid = tracker_cfg.DO_REID
        self.max_features_num = tracker_cfg.MAX_FEATURES_NUM
        self.reid_sim_threshold = tracker_cfg.REID_SIM_THRESHOLD  # lower is stricter
        self.reid_iou_threshold = tracker_cfg.REID_IOU_THRESHOLD  # optional geometric gate

        # Image alignment and motion-model controls
        self.do_align = tracker_cfg.DO_ALIGN
        self.motion_model_cfg = tracker_cfg.MOTION_MODEL

        # ECC alignment configuration (OpenCV's findTransformECC)
        self.warp_mode = eval(tracker_cfg.WARP_MODE)
        self.number_of_iterations = tracker_cfg.NUMBER_OF_ITERATIONS
        self.termination_eps = tracker_cfg.TERMINATION_EPS

        # Internal state
        self.tracks = []
        self.inactive_tracks = []
        self.track_num = 0
        self.im_index = 0
        # results[track_id][frame_idx] = [x1,y1,x2,y2,score]
        self.results = {}

    def reset(self, hard=True):
        """Reset tracker state; if hard=True also reset counters and stored results."""
        self.tracks = []
        self.inactive_tracks = []

        if hard:
            self.track_num = 0
            self.results = {}
            self.im_index = 0

    def tracks_to_inactive(self, tracks):
        """
        Move given tracks from active to inactive, freezing their last confident position.
        This preserves state for possible ReID-based re-activation in future frames.
        """
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features):
        """Initializes new Track objects and saves them (from remaining detections)."""
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            self.tracks.append(
                Track(
                    new_det_pos[i].view(1, -1),
                    new_det_scores[i],
                    self.track_num + i,
                    new_det_features[i].view(1, -1),
                    self.inactive_patience,
                    self.max_features_num,
                    (
                        self.motion_model_cfg.N_STEPS
                        if self.motion_model_cfg.N_STEPS > 0
                        else 1
                    ),
                )
            )
        self.track_num += num_new

    def regress_tracks(self, blob):
        """
        Regress positions of active tracks via the detector head and check keep-alive scores.
        Tracks falling below REGRESSION_PERSON_THRESH are deactivated.

        Returns:
            Tensor[K]: confidence scores for the *kept* regressed tracks (CUDA).
        """
        pos = self.get_pos()

        # Regress boxes for current frame features; clip to image bounds.
        boxes, scores = self.obj_detect.predict_boxes(pos, blob["features"])
        pos = clip_boxes_to_image(boxes, blob["img"].shape[-2:])

        s = []
        # Update in reverse to allow safe removal when deactivating
        for i in range(len(self.tracks) - 1, -1, -1):
            t = self.tracks[i]
            t.score = scores[i]
            if scores[i] <= self.regression_person_thresh:
                # Below keep-alive -> move to inactive
                self.tracks_to_inactive([t])
            else:
                s.append(scores[i])
                # Keep the regressed, clipped box
                # t.prev_pos = t.pos
                t.pos = pos[i].view(1, -1)

        # Scores are reversed due to reverse iteration
        return torch.Tensor(s[::-1]).cuda()

    def get_pos(self):
        """Get stacked positions of all active tracks as a Tensor[M,4] (may be empty)."""
        if len(self.tracks) == 1:
            pos = self.tracks[0].pos
        elif len(self.tracks) > 1:
            pos = torch.cat([t.pos for t in self.tracks], 0)
        else:
            pos = torch.zeros(0).cuda()
        return pos

    def get_features(self):
        """Get stacked appearance features of all active tracks (may be empty)."""
        if len(self.tracks) == 1:
            features = self.tracks[0].features
        elif len(self.tracks) > 1:
            features = torch.cat([t.features for t in self.tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def get_inactive_features(self):
        """Get stacked appearance features of all inactive tracks (may be empty)."""
        if len(self.inactive_tracks) == 1:
            features = self.inactive_tracks[0].features
        elif len(self.inactive_tracks) > 1:
            features = torch.cat([t.features for t in self.inactive_tracks], 0)
        else:
            features = torch.zeros(0).cuda()
        return features

    def reid(self, blob, new_det_pos, new_det_scores):
        """
        Attempt to ReID inactive tracks with new detections using appearance features
        (feature queues averaged per track) and Hungarian assignment. Optionally gate
        candidates by IoU to avoid implausible matches.

        Returns:
            (new_det_pos, new_det_scores, new_det_features) possibly reduced by assignments.
        """
        new_det_features = [torch.zeros(0).cuda() for _ in range(len(new_det_pos))]

        if self.do_reid:
            # Extract appearance embeddings for candidate detections
            new_det_features = self.get_appearances(blob, new_det_pos)

            if len(self.inactive_tracks) >= 1:
                # Build distance matrix (inactive_tracks x detections)
                dist_mat, pos = [], []
                for t in self.inactive_tracks:
                    dist_mat.append(
                        torch.cat(
                            [
                                t.test_features(feat.view(1, -1))
                                for feat in new_det_features
                            ],
                            dim=1,
                        )
                    )
                    pos.append(t.pos)
                if len(dist_mat) > 1:
                    dist_mat = torch.cat(dist_mat, 0)
                    pos = torch.cat(pos, 0)
                else:
                    dist_mat = dist_mat[0]
                    pos = pos[0]

                # Optional IoU gating: disallow matches with low overlap
                if self.reid_iou_threshold:
                    iou = bbox_overlaps(pos, new_det_pos)
                    iou_mask = torch.ge(iou, self.reid_iou_threshold)
                    iou_neg_mask = ~iou_mask
                    # Penalize impossible assignments heavily
                    dist_mat = dist_mat * iou_mask.float() + iou_neg_mask.float() * 1000
                dist_mat = dist_mat.cpu().numpy()

                # Solve assignment (Hungarian); accept if distance <= threshold
                row_ind, col_ind = linear_sum_assignment(dist_mat)

                assigned = []
                remove_inactive = []
                for r, c in zip(row_ind, col_ind):
                    if dist_mat[r, c] <= self.reid_sim_threshold:
                        t = self.inactive_tracks[r]
                        self.tracks.append(t)
                        t.count_inactive = 0
                        t.pos = new_det_pos[c].view(1, -1)
                        t.reset_last_pos()
                        t.add_features(new_det_features[c].view(1, -1))
                        assigned.append(c)
                        remove_inactive.append(t)

                # Remove re-activated from inactive pool
                for t in remove_inactive:
                    self.inactive_tracks.remove(t)

                # Keep only detections that were NOT consumed by re-activation
                keep = (
                    torch.Tensor(
                        [i for i in range(new_det_pos.size(0)) if i not in assigned]
                    )
                    .long()
                    .cuda()
                )
                if keep.nelement() > 0:
                    new_det_pos = new_det_pos[keep]
                    new_det_scores = new_det_scores[keep]
                    new_det_features = new_det_features[keep]
                else:
                    # Everything matched; return empty sets
                    new_det_pos = torch.zeros(0).cuda()
                    new_det_scores = torch.zeros(0).cuda()
                    new_det_features = torch.zeros(0).cuda()

        return new_det_pos, new_det_scores, new_det_features

    def get_appearances(self, blob, pos):
        """Compute ReID embeddings for given boxes using the siamese CNN."""
        new_features = self.reid_network.test_rois(blob["img"], pos).data
        return new_features

    def add_features(self, new_features):
        """Append fresh embeddings to each active track's feature queue (with cap)."""
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))

    def align(self, blob):
        """
        ECC-based global alignment to compensate inter-frame camera motion.
        Warps positions (active & inactive) and historical positions if motion model
        is enabled, so regression and NMS happen in the current frame's coordinates.
        """
        if self.im_index > 0:
            im1 = np.transpose(self.last_image.cpu().numpy(), (1, 2, 0))
            im2 = np.transpose(blob["img"][0].cpu().numpy(), (1, 2, 0))
            im1_gray = cv2.cvtColor(im1, cv2.COLOR_RGB2GRAY)
            im2_gray = cv2.cvtColor(im2, cv2.COLOR_RGB2GRAY)
            warp_matrix = np.eye(2, 3, dtype=np.float32)
            criteria = (
                cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
                self.number_of_iterations,
                self.termination_eps,
            )

            # Estimate 2x3 warp (e.g., affine) mapping im1->im2
            cc, warp_matrix = cv2.findTransformECC(
                im1_gray, im2_gray, warp_matrix, self.warp_mode, criteria
            )
            warp_matrix = torch.from_numpy(warp_matrix)

            # Warp current active tracks
            for t in self.tracks:
                t.pos = warp_pos(t.pos, warp_matrix)
                # t.pos = clip_boxes(Variable(pos), blob['im_info'][0][:2]).data

            # Warp inactive for potential ReID matching in the right coords
            if self.do_reid:
                for t in self.inactive_tracks:
                    t.pos = warp_pos(t.pos, warp_matrix)

            # Keep history consistent with the new frame's coordinates
            if self.motion_model_cfg.ENABLED:
                for t in self.tracks:
                    for i in range(len(t.last_pos)):
                        t.last_pos[i] = warp_pos(t.last_pos[i], warp_matrix)

    def motion_step(self, track):
        """Advance a track one step using last average velocity (center-only or full box)."""
        if self.motion_model_cfg.CENTER_ONLY:
            center_new = get_center(track.pos) + track.last_v
            track.pos = make_pos(
                *center_new, get_width(track.pos), get_height(track.pos)
            )
        else:
            track.pos = track.pos + track.last_v

    def motion(self):
        """
        Apply a simple linear motion model using the last n_steps positions.
        Computes average velocity across consecutive historical positions and
        updates the track once; inactive tracks can also be advanced if allowed.
        """
        for t in self.tracks:
            last_pos = list(t.last_pos)

            # Average velocity across history window (center-only or full box deltas)
            if self.motion_model_cfg.CENTER_ONLY:
                vs = [
                    get_center(p2) - get_center(p1)
                    for p1, p2 in zip(last_pos, last_pos[1:])
                ]
            else:
                vs = [p2 - p1 for p1, p2 in zip(last_pos, last_pos[1:])]

            t.last_v = torch.stack(vs).mean(dim=0)
            self.motion_step(t)

        if self.do_reid:
            for t in self.inactive_tracks:
                if t.last_v.nelement() > 0:
                    self.motion_step(t)

    def step(self, blob, dets):
        """Run one full tracking step on the current frame (image + detections)."""
        for t in self.tracks:
            # Append current position to history for motion averaging & stability
            t.last_pos.append(t.pos.clone())

        ###########################
        # Look for new detections #
        ###########################

        # self.obj_detect.load_image(blob['img'])

        # Choose public detections (if provided) or detector outputs
        if self.public_detections:
            dets = blob["dets"].squeeze(dim=0)
            if dets.nelement() > 0:
                boxes, scores = dets["boxes"], dets["scores"]
            else:
                boxes = scores = torch.zeros(0).cuda()
        else:
            boxes, scores = dets["boxes"], dets["scores"]

        if boxes.nelement() > 0:
            boxes = clip_boxes_to_image(boxes, blob["img"].shape[-2:])

            # Gate detections by score threshold (new-track candidate gate)
            inds = (
                torch.gt(scores, self.detection_person_thresh)
                .nonzero(as_tuple=False)
                .view(-1)
            )
        else:
            inds = torch.zeros(0).cuda()

        if inds.nelement() > 0:
            det_pos = boxes[inds]

            det_scores = scores[inds]
        else:
            det_pos = torch.zeros(0).cuda()
            det_scores = torch.zeros(0).cuda()

        ##################
        # Predict tracks #
        ##################

        if len(self.tracks):
            # Align positions to current frame using ECC (helps when camera moves)
            if self.do_align:
                self.align(blob)

            # Extrapolate using simple linear motion (optional)
            if self.motion_model_cfg.ENABLED:
                self.motion()
                # Remove degenerate boxes produced by motion extrapolation
                self.tracks = [t for t in self.tracks if t.has_positive_area()]

            # Regress track boxes on current frame; may deactivate weak tracks
            person_scores = self.regress_tracks(blob)

            if len(self.tracks):
                # NMS among regressed tracks to remove overlapping duplicates
                keep = nms(self.get_pos(), person_scores, self.regression_nms_thresh)

                self.tracks_to_inactive(
                    [
                        self.tracks[i]
                        for i in list(range(len(self.tracks)))
                        if i not in keep
                    ]
                )

                # Refresh appearance features for survivors (if using ReID)
                if keep.nelement() > 0 and self.do_reid:
                    new_features = self.get_appearances(blob, self.get_pos())
                    self.add_features(new_features)

        #####################
        # Create new tracks #
        #####################

        # !!! Here NMS is used to filter out detections that are already covered by tracks. This is
        # !!! done by iterating through the active tracks one by one, assigning them a bigger score
        # !!! than 1 (maximum score for detections) and then filtering the detections with NMS.
        # !!! In the paper this is done by calculating the overlap with existing tracks, but the
        # !!! result stays the same.
        if det_pos.nelement() > 0:
            keep = nms(det_pos, det_scores, self.detection_nms_thresh)
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

            # Iterate through tracks once; remove detections suppressed by each track
            # (Note: tracks could theoretically suppress each other if chained.)
            for t in self.tracks:
                nms_track_pos = torch.cat([t.pos, det_pos])
                nms_track_scores = torch.cat(
                    [torch.tensor([2.0]).to(det_scores.device), det_scores]
                )
                keep = nms(nms_track_pos, nms_track_scores, self.detection_nms_thresh)

                keep = keep[torch.ge(keep, 1)] - 1

                det_pos = det_pos[keep]
                det_scores = det_scores[keep]
                if keep.nelement() == 0:
                    break

        if det_pos.nelement() > 0:
            new_det_pos = det_pos
            new_det_scores = det_scores

            # Try to re-identify inactive tracks first; remaining become new tracks
            new_det_pos, new_det_scores, new_det_features = self.reid(
                blob, new_det_pos, new_det_scores
            )

            # Add new tracks from unmatched detections
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, new_det_features)

        ####################
        # Generate Results #
        ####################

        # Save per-id, per-frame results as [x1,y1,x2,y2,score]
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate(
                [t.pos[0].cpu().numpy(), np.array([t.score.cpu()])]
            )

        # Age inactive tracks; prune if degenerate or over patience
        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t
            for t in self.inactive_tracks
            if t.has_positive_area() and t.count_inactive <= self.inactive_patience
        ]

        # Advance frame index and store last image for ECC alignment
        self.im_index += 1
        self.last_image = blob["img"][0]

    def get_results(self):
        """Return the nested dict: results[track_id][frame_idx] -> np.ndarray(5,)."""
        return self.results


class Track(object):
    """Container for a single track's state, features, and short motion history."""

    def __init__(
        self,
        pos,
        score,
        track_id,
        features,
        inactive_patience,
        max_features_num,
        mm_steps,
    ):
        self.id = track_id
        self.pos = pos
        self.score = score
        # FIFO queue of per-frame embeddings; averaged at test-time for matching
        self.features = deque([features])
        self.ims = deque([])
        self.count_inactive = 0
        self.inactive_patience = inactive_patience
        self.max_features_num = max_features_num
        # Keep last (mm_steps + 1) positions for velocity averaging
        self.last_pos = deque([pos.clone()], maxlen=mm_steps + 1)
        self.last_v = torch.Tensor([])
        self.gt_id = None

    def has_positive_area(self):
        """Reject boxes with non-positive area (degenerate after warp/motion)."""
        return self.pos[0, 2] > self.pos[0, 0] and self.pos[0, 3] > self.pos[0, 1]

    def add_features(self, features):
        """Add a new embedding to the queue, pruning the oldest if over capacity."""
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """
        Compute appearance distance between this track (mean-pooled features)
        and the given test feature(s). Lower distance = better match.
        """
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]
        features = features.mean(0, keepdim=True)
        # dist = metrics.compute_distance_matrix(features, test_features)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist

    def reset_last_pos(self):
        """Reset short motion history to the current position (after re-activation)."""
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())
