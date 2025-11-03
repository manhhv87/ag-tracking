import numpy as np
from collections import deque
from scipy.optimize import linear_sum_assignment

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou
from torchvision.ops.boxes import clip_boxes_to_image, nms


class Tracker:
    """
    Orchestrates per-frame multi-object tracking with optional ReID.

    Parameters
    ----------
    detector : object
        Detector providing `predict_boxes(pos, features, class_id)`.
    reid_network : object
        Embedding extractor with `.test_rois(img, boxes)` returning per-ROI features.
    tracker_cfg : object / AttrDict
        Configuration holding thresholds and toggles:
        - DETECTION_PERSON_THRESH, REGRESSION_PERSON_THRESH, NMS_THRESH
        - DO_REID, INACTIVE_PATIENCE, MAX_FEATURES_NUM, REID_SIM_THRESHOLD
    """

    def __init__(self, detector, reid_network, tracker_cfg):
        # External components
        self.detector = detector
        self.reid_network = reid_network

        # Thresholds / hyperparameters from config
        self.detection_person_thresh = tracker_cfg.DETECTION_PERSON_THRESH
        self.regression_person_thresh = tracker_cfg.REGRESSION_PERSON_THRESH
        self.nms_thresh = tracker_cfg.NMS_THRESH

        self.do_reid = tracker_cfg.DO_REID
        self.inactive_patience = tracker_cfg.INACTIVE_PATIENCE
        self.max_features_num = tracker_cfg.MAX_FEATURES_NUM
        self.reid_sim_threshold = tracker_cfg.REID_SIM_THRESHOLD

        # State
        self.tracks = []            # active Track instances
        self.inactive_tracks = []   # recently lost tracks (eligible for ReID)
        self.track_num = 0          # global, ever-increasing track ID counter
        self.im_index = 0           # current frame index (0-based)
        self.results = {}           # {track_id: {frame_idx: [x1,y1,x2,y2,score]}}

    def reset(self, hard=True):
        """
        Reset tracker state.

        Parameters
        ----------
        hard : bool
            If True, also reset global counters and results. If False, only clear
            active/inactive lists (e.g., for scene cuts while preserving IDs).
        """
        self.tracks = []
        self.inactive_tracks = []
        if hard:
            self.track_num = 0
            self.im_index = 0
            self.results = {}

    def tracks_to_inactive(self, tracks):
        """
        Move given active tracks to the inactive pool, freezing their last position.

        Parameters
        ----------
        tracks : list[Track]
            Subset of currently active tracks to deactivate.
        """
        self.tracks = [t for t in self.tracks if t not in tracks]
        for t in tracks:
            t.pos = t.last_pos[-1]  # freeze at most recent recorded bbox
        self.inactive_tracks += tracks

    def add(self, new_det_pos, new_det_scores, new_det_features, hw):
        """Initializes new Track objects and saves them.

        Parameters
        ----------
        new_det_pos : Tensor[N,4]
            New detection boxes (x1,y1,x2,y2) to spawn tracks from.
        new_det_scores : Tensor[N]
            Detection confidences.
        new_det_features : Tensor[N,D] or empty
            Optional appearance embeddings to seed the track features queue.
        hw : tuple(int,int)
            Image size as (H, W); used by Track geometry checks.

        Notes
        -----
        - Spawns only if the box has positive area and is close to edges (heuristic),
          or if this is the very first frame (bootstrap all).
        """
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            track_num = self.track_num + 1
            track = Track(
                new_det_pos[i].view(1, -1),
                new_det_scores[i],
                track_num,
                new_det_features[i].view(1, -1),
                self.max_features_num,
                hw,
            )

            # Edge/area gating or unconditional on first frame
            if track.has_positive_area() and track.isin_edge() or self.im_index == 0:
                self.tracks.append(track)
                self.track_num = track_num

    def regress_tracks(self, blob):
        """Regress the position of the tracks and also checks their scores.

        Parameters
        ----------
        blob : dict
            A package holding:
            - "features": backbone features from the detector
            - "img": original image tensor (C,H,W) inside `blob["img"]`
        """
        if len(self.tracks):
            # Gather stacked positions for all active tracks
            pos, _ = self.get_pos()

            # Regress/refine positions & scores from ROI heads (class id fixed to 1)
            boxes, scores = self.detector.predict_boxes(pos, blob["features"], 1)

            # Clamp to image bounds for safety
            pos = clip_boxes_to_image(boxes, blob["img"].shape[-2:])

            # Update each track in-place; drop those that score to zero
            for i in range(len(self.tracks) - 1, -1, -1):
                t = self.tracks[i]
                t.score = scores[i]
                t.pos = pos[i].view(1, -1)
                if scores[i] == 0:
                    self.tracks_to_inactive([t])

    def get_pos(self):
        """Get the positions of all active tracks.

        Returns
        -------
        pos : Tensor[K,4] or Tensor[0]
            Stacked boxes for K active tracks (or empty).
        score : Tensor[K] or Tensor[0]
            Per-track current scores.
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

    def reid(self, blob, new_det_pos, new_det_scores):
        """Tries to ReID inactive tracks with provided detections.

        Parameters
        ----------
        blob : dict
            Must contain:
            - "img": tensor used by `reid_network.test_rois`
        new_det_pos : Tensor[M,4]
            Candidate detection boxes.
        new_det_scores : Tensor[M]
            Candidate detection confidences.

        Returns
        -------
        new_det_pos, new_det_scores, new_det_features : Tensors
            Remaining unmatched detections (after possible ReID matches) and their
            features (if DO_REID), to be used to spawn *new* tracks.

        Notes
        -----
        - When DO_REID is False, new_det_features are zero tensors and inputs are
          passed through unchanged.
        - When DO_REID is True:
          * Compute embeddings for all new detections.
          * For each inactive track, compute distances to all detection embeddings
            using `Track.test_features` (averaged pairwise distances).
          * Solve a linear assignment on the distance matrix and revive pairs with
            distance <= REID_SIM_THRESHOLD.
          * Remove revived detections from the pool before spawning new tracks.
        """
        new_det_features = [torch.zeros(0).cuda() for _ in range(len(new_det_pos))]

        if self.do_reid:
            # Extract appearance features for candidate detections
            new_det_features = self.reid_network.test_rois(
                blob["img"], new_det_pos
            ).data

            if len(self.inactive_tracks) >= 1:
                # Build distance matrix: rows=inactive tracks, cols=new dets
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
                    
                if len(dist_mat) > 1:
                    dist_mat = torch.cat(dist_mat, 0)
                else:
                    dist_mat = dist_mat[0]

                # Hungarian assignment on CPU numpy array
                dist_mat = dist_mat.cpu().numpy()
                row_ind, col_ind = linear_sum_assignment(dist_mat)
                assigned = []
                remove_inactive = []
                for r, c in zip(row_ind, col_ind):
                    if dist_mat[r, c] <= self.reid_sim_threshold:
                        # Reconnect: move inactive->active, reset counters/pos/features
                        t = self.inactive_tracks[r]
                        self.tracks.append(t)
                        t.count_inactive = 0
                        t.pos = new_det_pos[c].view(1, -1)
                        t.reset_last_pos()
                        t.add_features(new_det_features[c].view(1, -1))
                        assigned.append(c)
                        remove_inactive.append(t)

                # # BH DEBUG: collecting ReID distances
                # with open('output/reids/dists.txt', 'a') as logger:
                # 	dists = dist_mat.flatten()
                # 	dists = dists[dists < 999]
                # 	if len(dists): logger.write('\n'.join(['%6.3f' % num for num in dists])+ '\n')
                # # BH DEBUG END

                # Remove successfully re-identified tracks from the inactive pool
                for t in remove_inactive:
                    self.inactive_tracks.remove(t)

                # Keep only detections that were NOT assigned during ReID
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
                    # No remaining unmatched detections
                    new_det_pos = torch.zeros(0).cuda()
                    new_det_scores = torch.zeros(0).cuda()
                    new_det_features = torch.zeros(0).cuda()

        return new_det_pos, new_det_scores, new_det_features

    def add_features(self, new_features):
        """Adds new appearance features to active tracks.

        Parameters
        ----------
        new_features : Tensor[K,D]
            Features aligned with current active tracks (same order as `self.tracks`).
        """
        for t, f in zip(self.tracks, new_features):
            t.add_features(f.view(1, -1))

    def step(self, blob, detections):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.

        Workflow
        --------
        1) Append current positions to `last_pos` of each active track.
        2) Filter detections by class (keep class_id == 1).
        3) Regress track positions with detector ROI heads.
        4) Fuse detections + tracks using a combined NMS pass:
           - detections may replace overlapping tracks with higher score,
           - otherwise retain tracks/detections by per-branch thresholds.
        5) (Optional) ReID loop to revive inactive tracks, then spawn new tracks.
        6) Export per-frame results and update inactivity counters.
        """
        for t in self.tracks:
            # Keep a rolling history of positions for smoothing/revival
            # (deque lives in Track; we store a clone to freeze this frame's value)
            t.last_pos.append(t.pos.clone())

        # ---------------------------------------------------------------------
        # 1) Class-filter incoming detections (keep only class id == 1)
        # ---------------------------------------------------------------------
        boxes, scores, labels = (
            detections["boxes"],
            detections["scores"],
            detections["labels"],
        )
        det_pos = boxes[labels == 1]  # class id
        det_scores = scores[
            labels == 1
        ]  # class id                                       

        # ---------------------------------------------------------------------
        # 2) Predict (regress) active tracks using detector ROI heads
        # ---------------------------------------------------------------------
        if len(self.tracks):
            # regress new bbox locations
            self.regress_tracks(blob)

        # ---------------------------------------------------------------------
        # 3) Fuse detections and tracks with a combined NMS strategy
        # ---------------------------------------------------------------------
        if det_pos.nelement() > 0 and len(self.tracks) > 0:
            # Concatenate positions and scores: first tracks, then detections
            pos, score = self.get_pos()
            poses = torch.cat([pos, det_pos])
            scores = torch.cat([score, det_scores])

            # Track indices/types aligned with `poses`
            pos_idxs = list(range(len(self.tracks))) + list(range(det_pos.shape[0]))
            pos_type = len(self.tracks) * ["t"] + det_pos.shape[0] * ["d"]
            
            # IoU matrix and vanilla NMS to select survivors
            ious = box_iou(poses, poses)
            nms_keep = nms(poses, scores, self.nms_thresh)

            # Keep masks for detections and tracks
            det_pos_keep = torch.zeros(det_pos.shape[0]).cuda().bool()
            trk_pos_keep = torch.zeros(len(self.tracks)).cuda().bool()

            for idx in nms_keep:
                if pos_type[idx] == "d":    # detection wins
                    replaced = False        # whether it replaced a track
                    supress = torch.gt(ious[idx], self.nms_thresh)
                    indexes, lens = [], []
                    for i, sup in enumerate(supress):
                        # If this detection suppresses a track (overlaps strongly)
                        if sup.item() and i != idx.item() and pos_type[i] == "t":
                            replaced = True
                            # Update that track's box/score by the detection's
                            self.tracks[pos_idxs[i]].pos = poses[idx].view(1, -1)
                            self.tracks[pos_idxs[i]].score = scores[idx]  # and score
                            indexes.append(pos_idxs[i]) # collect indexes
                            # Keep the track that has the longest history (more stable)
                            lens.append(len(self.tracks[indexes[-1]].last_pos))
                    
                    # Mark exactly one replaced track (the longest-history one) to keep
                    if replaced:
                        trk_pos_keep[indexes[lens.index(max(lens))]] = True
                    
                    # If it didn't replace a track, consider spawning it if score high enough
                    if (
                        not replaced
                        and scores[idx].item() > self.detection_person_thresh
                    ):
                        det_pos_keep[pos_idxs[idx]] = True
                else:  # surviving track from NMS
                    if scores[idx].item() > self.regression_person_thresh:
                        trk_pos_keep[pos_idxs[idx]] = True

            # Keep only selected detections; deactivate tracks that did not survive thresholds
            det_pos = det_pos[det_pos_keep]
            det_scores = det_scores[det_pos_keep]
            self.tracks_to_inactive(
                [t for i, t in enumerate(self.tracks) if not trk_pos_keep[i].item()]
            )

            # For ReID: refresh features of remaining active tracks
            if len(self.tracks) > 0 and self.do_reid:
                pos, _ = self.get_pos()
                new_features = self.reid_network.test_rois(blob["img"], pos).data
                self.add_features(new_features)

        elif det_pos.nelement() > 0:
            # No active tracks: NMS within detections only to filter duplicates
            keep = nms(det_pos, det_scores, self.nms_thresh)
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

        # ---------------------------------------------------------------------
        # 4) ReID revival + spawn new tracks from remaining detections
        # ---------------------------------------------------------------------
        if det_pos.nelement() > 0:
            new_det_pos = det_pos
            new_det_scores = det_scores

            # Attempt to revive inactive tracks via appearance matching
            new_det_pos, new_det_scores, new_det_features = self.reid(
                blob, new_det_pos, new_det_scores
            )

            # Spawn new tracks for still-unmatched detections
            if new_det_pos.nelement() > 0:
                self.add(
                    new_det_pos,
                    new_det_scores,
                    new_det_features,
                    (blob["img"].shape[2], blob["img"].shape[3]),
                )

        # ---------------------------------------------------------------------
        # 5) Export this frame's results and update inactivity counters
        # ---------------------------------------------------------------------
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}

            # Store [x1,y1,x2,y2,score] per frame for this track
            self.results[t.id][self.im_index] = np.concatenate(
                [t.pos[0].cpu(), np.array([t.score.cpu()])]
            )

        # Age inactive tracks; keep only geometrically valid ones
        for t in self.inactive_tracks:
            t.count_inactive += 1

        self.inactive_tracks = [
            t for t in self.inactive_tracks if t.has_positive_area()
        ]

        # Advance global frame index
        self.im_index += 1

    def get_results(self):
        """
        Return the accumulated MOT-format results.

        Returns
        -------
        dict
            Nested dict: {track_id: {frame_idx: np.array([x1,y1,x2,y2,score])}}
        """
        return self.results


def isin_edge(box, hw):
    """
    Heuristic to detect whether a box lies close to an image border.

    Parameters
    ----------
    box : Tensor[1,4]
        Single box (x1,y1,x2,y2).
    hw : tuple(int,int)
        Image size (H, W).

    Returns
    -------
    str or bool
        One of {'r','t','l','b'} for right/top/left/bottom proximity, or False
        when not near an edge.
    """
    edge = False
    if box[0, 2] - box[0, 0] > (hw[1] - box[0, 2]) * 2:
        edge = "r"
    elif box[0, 3] - box[0, 1] > box[0, 1] * 2:
        edge = "t"
    elif box[0, 2] - box[0, 0] > box[0, 0] * 2:
        edge = "l"
    elif box[0, 3] - box[0, 1] > (hw[0] - box[0, 3]) * 2:
        edge = "b"
    return edge


class Track(object):
    """This class contains all necessary for every individual track.

    Parameters
    ----------
    pos : Tensor[1,4]
        Initial bbox (x1,y1,x2,y2).
    score : Tensor[] or scalar-like
        Initial confidence score.
    track_id : int
        Unique identifier.
    features : Tensor[1,D]
        Initial appearance features (can be empty tensor).
    max_features_num : int
        Maximum queue length for stored features.
    hw : tuple(int,int)
        Image size (H, W) used by geometry checks.
    back : int, default 5
        History length for `last_pos` deque.
    """

    def __init__(self, pos, score, track_id, features, max_features_num, hw, back=5):
        self.hw = hw
        self.id = track_id
        self.pos = pos
        self.score = score
        self.features = deque([features])   # queue of feature tensors
        self.count_inactive = 0
        self.max_features_num = max_features_num
        self.last_pos = deque([pos.clone()], maxlen=back)   # recent positions
        self.last_v = torch.Tensor([])  # (optional) velocity placeholder

    def has_positive_area(self):
        """
        Check that the box is big enough relative to the image.

        Returns
        -------
        bool
            True iff width and height exceed ~1% of image size (heuristic).
        """
        # is x2 > x1 + target and y2 > y1 + target
        size_x = self.pos[0, 2] > self.pos[0, 0] + self.hw[1] / (100 / 1)  # percentage
        size_y = self.pos[0, 3] > self.pos[0, 1] + self.hw[0] / (100 / 1)  # percentage
        return size_x and size_y

    def isin_edge(self):
        """
        Cache and return the edge flag for the current box.

        Returns
        -------
        str or bool
            Same semantics as module-level `isin_edge`.
        """
        self.edge = isin_edge(self.pos, self.hw)
        return self.edge

    def add_features(self, features):
        """Adds new appearance features to the object.

        Parameters
        ----------
        features : Tensor[1,D]
            New embedding vector to append to the feature queue.
        """
        self.features.append(features)
        if len(self.features) > self.max_features_num:
            self.features.popleft()

    def test_features(self, test_features):
        """Compares test_features to features of this Track object

        Parameters
        ----------
        test_features : Tensor[1,D]
            Candidate embedding.

        Returns
        -------
        Tensor[1,1]
            Mean pairwise distance (smaller = more similar).
        """
        if len(self.features) > 1:
            features = torch.cat(list(self.features), dim=0)
        else:
            features = self.features[0]

        # features = features.mean(0, keepdim=True)
        dist = F.pairwise_distance(features, test_features, keepdim=True)
        return dist.mean(0, keepdim=True)
        # return dist

    def reset_last_pos(self):
        """
        Clear the recent position history and re-seed with the current bbox.
        """
        self.last_pos.clear()
        self.last_pos.append(self.pos.clone())
