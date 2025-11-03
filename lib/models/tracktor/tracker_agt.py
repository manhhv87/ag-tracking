import torch
import numpy as np
from scipy.optimize import linear_sum_assignment

from torchvision.ops import box_iou
from torchvision.ops.boxes import clip_boxes_to_image, nms

from .slam import SLAMProcess
from ...utils.geom_utils import estimate_euclidean_transform

torch.set_printoptions(sci_mode=False)


class Tracker:
    """
    Global-association Tracktor-style tracker (AGT variant).

    This tracker combines:
      1) **Track-by-detection regression** – existing tracks are refined each frame
         using the detector's ROI heads (via `detector.predict_boxes`).
      2) **Global spatial association** – a lightweight SLAM (LoFTR-based) process
         estimates per-frame camera motion so that centers of detections/tracks
         can be compared in a **global coordinate frame**. This enables re-linking
         inactive tracks (e.g., objects re-entering the FOV) using the Hungarian
         algorithm over Euclidean distances in the global frame.

    Why this mirrors the paper:
      • The paper's SAM uses detector-free local matching (LoFTR) to estimate a global
        motion and then performs association in global coordinates. Here, `SLAMProcess`
        provides the cumulative transform (H) and we convert centers into a consistent
        global frame for Hungarian assignment—functionally the same association idea.
      • Thresholds correspond to the paper's s_new (spawn), s_active (keep-alive),
        and λ_nms (NMS), whose robustness ranges are discussed in Fig. 2.

    Key attributes
    --------------
    detector : object
        Must implement `predict_boxes(pos, features, class_id)` -> (boxes, scores).
    slam : SLAMProcess
        Background process estimating cumulative transform H between frames.
    tracks : list[Track]
        Currently active tracks.
    inactive_tracks : list[Track]
        Recently lost tracks that can be revived by global association.
    cam_oxy : torch.Tensor (3x1)
        Camera origin (x, y, 1) in homogeneous coordinates extracted from SLAM H.
    track_num : int
        Monotonic increasing ID assigned to new tracks.
    im_index : int
        Current frame index (0-based).
    results : dict
        MOT-format results dictionary: {track_id: {frame_idx: [x1,y1,x2,y2,score]}}.
    """

    def __init__(self, detector, cfg, tracker_cfg):
        # External detector (e.g., Faster R-CNN + FPN wrapper) used to regress boxes
        self.detector = detector

        # Thresholds controlling NMS/keeping detections/tracks
        self.detection_person_thresh = tracker_cfg.DETECTION_PERSON_THRESH      # ~ s_new (spawn new track)
        self.regression_person_thresh = tracker_cfg.REGRESSION_PERSON_THRESH    # ~ s_active (keep-alive for regressed tracks)
        self.nms_thresh = tracker_cfg.NMS_THRESH                                # ~ λ_nms (overlap suppression)
        
        # Parameters for the SLAM-based global association module
        self.factor = tracker_cfg.REDUCTION_FACTOR    # downsample factor for LoFTR
        self.ransac = tracker_cfg.RANSAC_THRES        # RANSAC reprojection threshold
        
        # Spawn the SLAM child process (LoFTR matching + cumulative homography H)
        self.slam = SLAMProcess(cfg, self.factor, self.ransac)
        self.slam.start()

        # Tracking state containers
        self.tracks = []             # active tracks (updated each frame via regression/NMS)
        self.inactive_tracks = []    # tracks temporarily deactivated; eligible for revival

        # Camera origin in homogeneous coords; updated from SLAM's H each frame      
        self.cam_oxy = torch.tensor([[0], [0], [1]]).cuda().float()

        # Global counters and outputs
        self.track_num = 0           # last assigned track id (monotonic increasing)
        self.im_index = 0            # current frame index
        self.results = {}            # MOT output accumulator

    def reset(self, hard=True):
        """
        Reset the tracker's state.

        Parameters
        ----------
        hard : bool
            If True, clear **everything** including ID counters and results.
            If False, only clear current active/inactive track lists.

        Notes
        -----
        Hard reset is useful between sequences; soft reset can be used for segment runs.
        """
        self.tracks = []
        self.inactive_tracks = []
        if hard:
            self.track_num = 0
            self.im_index = 0
            self.results = {}

    def get_pos(self):
        """Get the positions of all active tracks.

        Returns
        -------
        (pos, score) : Tuple[Tensor, Tensor]
            - pos: stacked boxes (K, 4) for K active tracks, or an empty tensor.
            - score: stacked scores (K,) aligned with `pos`.

        Implementation detail
        ---------------------
        Uses CUDA tensors for downstream ops (box_iou/NMS). Fast path for 1 track.
        """
        if len(self.tracks) == 1:
            # Fast path for a single track
            pos = self.tracks[0].pos
            score = torch.Tensor([self.tracks[0].score.item()]).cuda()
        elif len(self.tracks) > 1:
            # Concatenate positions/scores of all tracks
            pos = torch.cat([t.pos for t in self.tracks], 0)
            score = torch.Tensor([t.score.item() for t in self.tracks]).cuda()
        else:
            # No active tracks
            pos = torch.zeros(0).cuda()
            score = torch.zeros(0).cuda()
        return pos, score

    def tracks_to_inactive(self, tracks):
        """
        Move a subset of active tracks into the inactive pool.

        Parameters
        ----------
        tracks : list[Track]
            The tracks to deactivate.

        Rationale
        ---------
        Mirrors the paper's inactive pool concept: temporarily remove uncertain tracks
        but keep their state for possible re-association once stronger evidence appears.
        """
        # Keep only tracks not in the removal set
        self.tracks = [t for t in self.tracks if t not in tracks]
        # Add them to inactive for potential re-identification later
        self.inactive_tracks += tracks

    def regress_tracks(self, blob):
        """Regress the position of the tracks and also checks their scores.

        Parameters
        ----------
        blob : dict
            Must include:
              - "features": backbone features needed by detector ROI heads,
              - "img": original image tensor for clipping boxes to image bounds.

        Behavior
        --------
        • Refines each active track's box with the detector head.
        • Clips to image bounds to avoid invalid geometry.
        • Deactivates tracks with near-zero scores (regression failure).
        """
        if len(self.tracks):
            # Gather current track positions; detector will refine them
            pos, _ = self.get_pos()
            boxes, scores = self.detector.predict_boxes(
                pos, blob["features"], 1
            )  # class id

            # Clip refined boxes to image size to avoid out-of-bounds
            pos = clip_boxes_to_image(boxes, blob["img"].shape[-2:])

            # Update track states in-place; deactivate if score ~ 0 (numerical eps)
            for i in range(len(self.tracks) - 1, -1, -1):
                t = self.tracks[i]
                t.score = scores[i]
                t.pos = pos[i].view(1, -1)
                if scores[i] < np.spacing(0):
                    self.tracks_to_inactive([t])

    def reid(self, det_pos, det_scores):
        """
        Re-identify tracks using global (SLAM) coordinates and Hungarian matching.

        This function attempts to:
          1) **Stabilize/refresh** the SLAM transform using active tracks whose
             centers are reliable (not near edges and already initialized).
          2) **Update `cam_oxy`** (camera origin) from the SLAM homography H.
          3) **Revive inactive tracks** by matching their global centers against
             globalized detection centers using `linear_sum_assignment`.

        Parameters
        ----------
        det_pos : Tensor[M, 4]
            Candidate detection boxes (class-filtered).
        det_scores : Tensor[M]
            Corresponding detection scores.

        Returns
        -------
        (det_pos, det_scores) : Tuple[Tensor, Tensor]
            Potentially reduced set of detections after removing those assigned
            to revived tracks. (Unassigned detections remain to spawn new tracks.)

        Notes
        -----
        • This is the counterpart of the paper's SAM association step: convert both
          sides (inactive tracks and detections) to a stable global frame, then solve
          a Hungarian assignment on pairwise Euclidean distances.
        • Distance gates: fixed 500 px threshold is used (can be replaced by a
          scale-aware threshold using t.rad to mimic radius-based gating).
        """
        # -------------------------------
        # A) Use active tracks to update SLAM transform if possible
        # -------------------------------
        if len(self.tracks):
            cxy_t, cxy_f = [], []   # target (previous global) vs. current frame centers
            for t in self.tracks:
                if (
                    not t.isin_edge() and t.cxy != None
                ):  # track center exists and is reliable (not on border)
                    cxy = (
                        t.pos[:, :2] + t.pos[:, 2:]
                    ) / 2  # center in the current frame (image coordinates)
                    cxy_f.append(cxy[0].tolist())       # frame t center
                    cxy_t.append(t.cxy[0].tolist())     # previously stored global center
            
            if len(cxy_f) > 1:  # only fit transform if we have multiple correspondences
                Ha = np.eye(3)
                # Estimate Euclidean transform (R|t) that maps current centers to previous global centers
                Ha[:2, :2], Ha[:2, 2] = estimate_euclidean_transform(
                    np.array(cxy_f), np.array(cxy_t)
                )
                self.slam.H = Ha    # overwrite SLAM H with this refined transform

            # Update camera origin (translation part of H)
            self.cam_oxy = torch.tensor(self.slam.H).cuda().float()[:, -1:]

            # Initialize global centers/radii for tracks that just became reliable
            for t in self.tracks:
                # not in edge first time (no cen and rad yet)
                if not t.isin_edge() and t.cxy == None:
                    # Global center = image center + camera translation (homography shift)
                    t.cxy = (t.pos[0, :2] + t.pos[0, 2:]) / 2 + self.cam_oxy[
                        0:2, 0
                    ].view(
                        1, 2
                    )  # center
                    # A proxy radius for scale gating (here: average of half-width and half-height)
                    t.rad = sum(t.pos[0, 2:] - t.pos[0, :2]) / 4  # radious
        else:
            # No active tracks – still refresh camera origin from SLAM
            self.cam_oxy = torch.tensor(self.slam.H).cuda().float()[:, -1:]

        # -------------------------------
        # B) Build candidate centers for inactive tracks (global coords)
        # -------------------------------
        cxy = []
        for t in self.inactive_tracks:
            if t.cxy != None:
                # Already have a cached global center
                cxy.append(t.cxy)
            else:
                # Fallback: use image center + camera origin cached at deactivation time
                cxy.append(
                    (t.pos[0, :2] + t.pos[0, 2:]) / 2 + t.cam_oxy[0:2, 0].view(1, 2)
                )

        # -------------------------------
        # C) Hungarian matching (inactive tracks vs. new detections) in global coords
        # -------------------------------
        if len(cxy):
            cxy = torch.cat(cxy)
            if det_pos.nelement() > 0:
                # Convert detection centers into the global frame using current cam translation
                det_cxy = (det_pos[:, :2] + det_pos[:, 2:]) / 2 + self.cam_oxy[
                    0:2, 0
                ].view(1, 2)

                # Cost = pairwise Euclidean distance between (inactive) and (detection) centers
                dist = torch.cdist(cxy[None, :, :], det_cxy[None, :, :])[0]

                # Solve optimal assignment on CPU (SciPy Hungarian)
                track_ind, det_ind = linear_sum_assignment(dist.cpu().numpy())

                not_assigned = []     # indices of detections that remain unassigned
                remove_inactive = []  # inactive tracks to revive (move to active)
                
                for tr, dt in zip(track_ind, det_ind):
                    # Current inactive track candidate and matched detection
                    t = self.inactive_tracks[tr]

                    # Distance gating: use a fixed threshold (500 px) or, if available,
                    # a scale-aware threshold based on the track's approximate radius.
                    if t.rad != None:
                        if dist[tr, dt] <= 500:  # 2 * t.rad:
                            self.tracks.append(t)
                            t.pos = det_pos[dt].view(1, -1)
                            remove_inactive.append(t)
                        else:
                            not_assigned.append(dt)
                    else:
                        if (
                            dist[tr, dt] <= 500
                        ):  # sum(det_pos[dt, 2:] - det_pos[dt, :2]) / 2:
                            self.tracks.append(t)
                            t.pos = det_pos[dt].view(1, -1)
                            remove_inactive.append(t)
                        else:
                            not_assigned.append(dt)
                
                # Remove revived tracks from the inactive pool
                for t in remove_inactive:
                    self.inactive_tracks.remove(t)
                
                # Keep only detections that were not assigned to revivals
                det_pos, det_scores = det_pos[not_assigned], det_scores[not_assigned]

        return det_pos, det_scores

    def add(self, new_det_pos, new_det_scores, hw):
        """Initializes new Track objects and saves them.

        Parameters
        ----------
        new_det_pos : Tensor[N, 4]
            Boxes (x1, y1, x2, y2) for new track seeds.
        new_det_scores : Tensor[N]
            Scores for the same detections.
        hw : Tuple[int, int]
            Image height/width; used for size/edge heuristics.

        Notes
        -----
        • Seeds include the current `cam_oxy` to immediately place an approximate
          global center if the box is not near an image edge (mirrors SAM's idea
          of maintaining global referential stability).
        • Spawn condition uses a positive-area & edge heuristic; on the first frame,
          spawning is allowed regardless to bootstrap IDs.
        """
        num_new = new_det_pos.size(0)
        for i in range(num_new):
            track_num = self.track_num + 1
            pos = new_det_pos[i].view(1, -1)
            score = new_det_scores[i]
            # Seed a new Track with current camera origin (cam_oxy) to place its global center
            track = Track(pos, self.cam_oxy, score, track_num, hw)
            # Spawn condition: box has positive area and lies near an image edge
            # (or force-create on the very first frame)
            if track.has_positive_area() and track.isin_edge() or self.im_index == 0:
                self.tracks.append(track)
                self.track_num = track_num

    def step(self, blob, detections):
        """This function should be called every timestep to perform tracking with a blob
        containing the image information.

        Pipeline
        --------
        1) Run SLAM on the current frame to update cumulative transform H.
        2) Filter detections by class (keep class_id == 1).
        3) Regress active tracks with detector ROI heads.
        4) Fuse tracks + detections with a combined NMS policy.
        5) ReID / global association to revive inactive tracks (Hungarian).
        6) Spawn new tracks from remaining detections.
        7) Log MOT-format results for this frame and increment `im_index`.

        Why it matches the paper:
        -------------------------
        Steps (4) and (5) correspond to the unified suppression and SAM-based global
        association respectively (Fig. 1). Threshold roles map to (s_new, s_active, λ_nms).
        """
        # ------------------------------------------------------------------
        # 1) SLAM step for this frame (runs LoFTR + RANSAC inside the process)
        # ------------------------------------------------------------------
        h, w = blob["img"].shape[2:4]            # (H, W) for SLAM downsampling
        self.slam.args = (blob["img"], h, w, self.im_index)
        self.slam.run()                          # call the process's run() entry (uses self.args)

        # ------------------------------------------------------------------
        # 2) Keep only detections of the target class (id == 1)
        # ------------------------------------------------------------------
        boxes, scores, labels = (
            detections["boxes"],
            detections["scores"],
            detections["labels"],
        )
        det_pos = boxes[labels == 1]        # class id
        det_scores = scores[labels == 1]    # class id

        # ------------------------------------------------------------------
        # 3) Predict (regress) active tracks to the current frame
        # ------------------------------------------------------------------
        if len(self.tracks):
            # regress new bbox locations
            self.regress_tracks(blob)

        # ------------------------------------------------------------------
        # 4) Combined NMS across {tracks + detections}
        #    - If a detection suppresses a track and has a higher score, we
        #      update the track with that detection's box/score.
        #    - Survivors are filtered by their respective thresholds.
        # ------------------------------------------------------------------
        if det_pos.nelement() > 0 and len(self.tracks) > 0:
            pos, score = self.get_pos()
            poses = torch.cat([pos, det_pos])        # concat track boxes + det boxes
            scores = torch.cat([score, det_scores])  # concat corresponding scores
            
            # Book-keeping arrays to map indices to "t"(rack) or "d"(etection)
            pos_idxs = list(range(len(self.tracks))) + list(range(det_pos.shape[0]))
            pos_type = len(self.tracks) * ["t"] + det_pos.shape[0] * ["d"]

            # IoU matrix and vanilla NMS to obtain kept indices
            ious = box_iou(poses, poses)
            nms_keep = nms(poses, scores, self.nms_thresh)

            # Keep masks initialized as False
            det_pos_keep = torch.zeros(det_pos.shape[0]).cuda().bool()
            trk_pos_keep = torch.zeros(len(self.tracks)).cuda().bool()

            for idx in nms_keep:
                if pos_type[idx] == "d":    # detection
                    replaced = False        # whether this detection replaced any track
                    supress = torch.gt(ious[idx], self.nms_thresh)
                    indexes, lens = [], []
                    for i, sup in enumerate(supress):
                        if (
                            sup.item() and i != idx.item() and pos_type[i] == "t"
                        ):  # if the detection supressed a track
                            replaced = True
                            # Adopt detection's box/score for the suppressed track
                            self.tracks[pos_idxs[i]].pos = poses[idx].view(1,-1)
                            self.tracks[pos_idxs[i]].score = scores[idx]  # and score
                            indexes.append(pos_idxs[i])  # collect indexes
                            # lens.append(len(self.tracks[indexes[-1]].last_pos))   # and lengths (global_pos)
                            lens.append(
                                scores[idx].item()
                            )  # temporarily changed to highest score
                    
                    # Keep only a single replaced track (e.g., the one with highest temp "lens")
                    if replaced:
                        trk_pos_keep[indexes[lens.index(max(lens))]] = True
                    
                    # Otherwise, accept this detection as a new seed if above threshold
                    if (
                        not replaced
                        and scores[idx].item() > self.detection_person_thresh
                    ):
                        det_pos_keep[pos_idxs[idx]] = True
                else:  # existing track
                    # Keep regressed track only if its score is above threshold
                    if scores[idx].item() > self.regression_person_thresh:
                        trk_pos_keep[pos_idxs[idx]] = True

            # Remove filtered detections; move filtered tracks to inactive
            det_pos = det_pos[det_pos_keep]
            det_scores = det_scores[det_pos_keep]
            self.tracks_to_inactive(
                [t for i, t in enumerate(self.tracks) if not trk_pos_keep[i].item()]
            )

        elif det_pos.nelement() > 0:
            # Fallback: NMS only among detections when there are no active tracks
            keep = nms(det_pos, det_scores, self.nms_thresh)
            det_pos = det_pos[keep]
            det_scores = det_scores[keep]

        # ------------------------------------------------------------------
        # 5) Re-identification / global association (Hungarian)
        # ------------------------------------------------------------------
        self.slam.join()  # ensure SLAM has finished the current step

        # Do ReID if activated
        if True:  # TODO: add param do-reid
            new_det_pos, new_det_scores = self.reid(det_pos, det_scores)

        # ------------------------------------------------------------------
        # 6) Create new tracks from the remaining detections
        # ------------------------------------------------------------------
        if len(new_det_pos):
            if new_det_pos.nelement() > 0:
                self.add(new_det_pos, new_det_scores, (h, w))

        # ------------------------------------------------------------------
        # 7) Log results for this frame
        # ------------------------------------------------------------------
        for t in self.tracks:
            if t.id not in self.results.keys():
                self.results[t.id] = {}
            self.results[t.id][self.im_index] = np.concatenate(
                [t.pos[0].cpu(), np.array([t.score.cpu()])]
            )
        self.im_index += 1

    def get_results(self):
        """
        Return accumulated tracking results in MOT format.

        Returns
        -------
        dict
            {track_id: {frame_idx: np.array([x1, y1, x2, y2, score])}}

        Usage
        -----
        Post-process this dict to write MOTChallenge text files or to compute
        HOTA/IDF1/MOTA as in the paper's evaluation.
        """
        return self.results


def isin_edge(box, hw):
    """
    Heuristic to classify whether a box lies near an image border.

    Parameters
    ----------
    box : Tensor[1, 4]
        Single bounding box (x1, y1, x2, y2).
    hw : Tuple[int, int]
        Image height and width.

    Returns
    -------
    str | bool
        One of {"r","t","l","b"} for right/top/left/bottom edges, or False.

    Notes
    -----
    This is used at spawn-time and when initializing global center/radius; boxes
    near borders are often unreliable for precise center estimation.
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

    A `Track` stores the current bounding box, score, ID, cached camera origin,
    and (optionally) a global center `cxy` and a scale proxy `rad`. The latter
    two are used by the global association stage to gate re-identification.

    Parameters
    ----------
    pos : Tensor[1, 4]
        Initial box in image coordinates.
    cam_oxy : Tensor[3, 1]
        Camera origin (homogeneous) at creation time, used to globalize centers.
    score : Tensor[] or scalar-like
        Initial detection score.
    track_id : int
        Unique identifier for the track.
    hw : Tuple[int, int]
        Image (height, width) for edge/area heuristics.
    """

    def __init__(self, pos, cam_oxy, score, track_id, hw):
        self.hw = hw
        self.pos = pos      # bounding box in the first frame coordinates (and subsequent frame coordinates)
        self.score = score  # score
        self.id = track_id  # ID
        self.cam_oxy = cam_oxy

        if not self.isin_edge():
            # Initialize global center using current image center + camera translation
            self.cxy = (self.pos[0, :2] + self.pos[0, 2:]) / 2 + cam_oxy[0:2, 0].view(1,2)
            # A coarse "radius" (average of half-width/half-height) for distance gating
            self.rad = sum(self.pos[0, 2:] - self.pos[0, :2]) / 4
        else:
            # Near border: do not trust centers/radii yet
            self.cxy = None
            self.rad = None

    def has_positive_area(self):
        """
        Check that the box is sufficiently large.

        Returns
        -------
        bool
            True if width and height exceed ~2% of W/H (heuristic).

        Rationale
        ---------
        Avoids initializing/tracking degenerate boxes (improves stability and
        aligns with paper's focus on robust identity over time).
        """
        # is x2 > x1 + target and y2 > y1 + target
        size_x = self.pos[0, 2] > self.pos[0, 0] + self.hw[1] / (100 / 2)  # percentage
        size_y = self.pos[0, 3] > self.pos[0, 1] + self.hw[0] / (100 / 2)  # percentage
        return size_x and size_y

    def isin_edge(self):
        """
        Determine whether the current box is near an image edge.

        Returns
        -------
        str | bool
            Same convention as the module-level `isin_edge`.

        Note
        ----
        Used to defer global-center initialization for unreliable edge boxes.
        """
        self.edge = isin_edge(self.pos, self.hw)
        return self.edge
