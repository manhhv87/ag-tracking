#########################################
# Still ugly file with helper functions #
#########################################

import os
from collections import defaultdict
from os import path as osp

import numpy as np
import torch
from cycler import cycler as cy

import cv2
import matplotlib
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
import motmetrics as mm
import pdb

# Use a non-interactive backend so plotting works on servers/headless environments.
matplotlib.use("Agg")

# https://matplotlib.org/cycler/
# get all colors with
# colors = []
# 	for name,_ in matplotlib.colors.cnames.items():
# 		colors.append(name)

# Color names (CSS palette) used for visualization only.
# These DO NOT affect tracking logic or metrics. Just for easier reading of plots.
colors = [
    "aliceblue",
    "antiquewhite",
    "aqua",
    "aquamarine",
    "azure",
    "beige",
    "bisque",
    "black",
    "blanchedalmond",
    "blue",
    "blueviolet",
    "brown",
    "burlywood",
    "cadetblue",
    "chartreuse",
    "chocolate",
    "coral",
    "cornflowerblue",
    "cornsilk",
    "crimson",
    "cyan",
    "darkblue",
    "darkcyan",
    "darkgoldenrod",
    "darkgray",
    "darkgreen",
    "darkgrey",
    "darkkhaki",
    "darkmagenta",
    "darkolivegreen",
    "darkorange",
    "darkorchid",
    "darkred",
    "darksalmon",
    "darkseagreen",
    "darkslateblue",
    "darkslategray",
    "darkslategrey",
    "darkturquoise",
    "darkviolet",
    "deeppink",
    "deepskyblue",
    "dimgray",
    "dimgrey",
    "dodgerblue",
    "firebrick",
    "floralwhite",
    "forestgreen",
    "fuchsia",
    "gainsboro",
    "ghostwhite",
    "gold",
    "goldenrod",
    "gray",
    "green",
    "greenyellow",
    "grey",
    "honeydew",
    "hotpink",
    "indianred",
    "indigo",
    "ivory",
    "khaki",
    "lavender",
    "lavenderblush",
    "lawngreen",
    "lemonchiffon",
    "lightblue",
    "lightcoral",
    "lightcyan",
    "lightgoldenrodyellow",
    "lightgray",
    "lightgreen",
    "lightgrey",
    "lightpink",
    "lightsalmon",
    "lightseagreen",
    "lightskyblue",
    "lightslategray",
    "lightslategrey",
    "lightsteelblue",
    "lightyellow",
    "lime",
    "limegreen",
    "linen",
    "magenta",
    "maroon",
    "mediumaquamarine",
    "mediumblue",
    "mediumorchid",
    "mediumpurple",
    "mediumseagreen",
    "mediumslateblue",
    "mediumspringgreen",
    "mediumturquoise",
    "mediumvioletred",
    "midnightblue",
    "mintcream",
    "mistyrose",
    "moccasin",
    "navajowhite",
    "navy",
    "oldlace",
    "olive",
    "olivedrab",
    "orange",
    "orangered",
    "orchid",
    "palegoldenrod",
    "palegreen",
    "paleturquoise",
    "palevioletred",
    "papayawhip",
    "peachpuff",
    "peru",
    "pink",
    "plum",
    "powderblue",
    "purple",
    "rebeccapurple",
    "red",
    "rosybrown",
    "royalblue",
    "saddlebrown",
    "salmon",
    "sandybrown",
    "seagreen",
    "seashell",
    "sienna",
    "silver",
    "skyblue",
    "slateblue",
    "slategray",
    "slategrey",
    "snow",
    "springgreen",
    "steelblue",
    "tan",
    "teal",
    "thistle",
    "tomato",
    "turquoise",
    "violet",
    "wheat",
    "white",
    "whitesmoke",
    "yellow",
    "yellowgreen",
]


# From frcnn/utils/bbox.py
def bbox_overlaps(boxes, query_boxes):
    """
    Compute the pairwise IoU (Intersection-over-Union) between two sets of boxes.

    Parameters
    ----------
    boxes : (N, 4) np.ndarray or torch.Tensor
        Each row is [x1, y1, x2, y2] (pixel coordinates).
    query_boxes : (K, 4) np.ndarray or torch.Tensor
        Each row is [x1, y1, x2, y2] to be compared against `boxes`.

    Returns
    -------
    overlaps : (N, K) same dtype as inputs
        The IoU matrix in [0, 1] for each (boxes_i, query_boxes_j).

    Notes
    -----
    - Supports both NumPy and PyTorch. Output matches the type of the input
      (if NumPy in, NumPy out).
    - Uses +1 for width/height as in the classic pixel-inclusive convention
      adopted by Faster R-CNN-style codebases.
    - IoU here is the fundamental measurement used by NMS/assignment heuristics.
    """

    if isinstance(boxes, np.ndarray):
        # Convert to torch for vectorized math, then convert back on return.
        boxes = torch.from_numpy(boxes)
        query_boxes = torch.from_numpy(query_boxes)
        out_fn = (
            lambda x: x.numpy()
        )  # If input is ndarray, turn the overlaps back to ndarray when return
    else:
        out_fn = lambda x: x

    # Areas with +1 convention.
    box_areas = (boxes[:, 2] - boxes[:, 0] + 1) * (boxes[:, 3] - boxes[:, 1] + 1)
    query_areas = (query_boxes[:, 2] - query_boxes[:, 0] + 1) * (
        query_boxes[:, 3] - query_boxes[:, 1] + 1
    )

    # Intersection width/height for all pairs (broadcasted).
    iw = (
        torch.min(boxes[:, 2:3], query_boxes[:, 2:3].t())
        - torch.max(boxes[:, 0:1], query_boxes[:, 0:1].t())
        + 1
    ).clamp(min=0)
    ih = (
        torch.min(boxes[:, 3:4], query_boxes[:, 3:4].t())
        - torch.max(boxes[:, 1:2], query_boxes[:, 1:2].t())
        + 1
    ).clamp(min=0)

    # Union = area(a) + area(b) - intersection
    ua = box_areas.view(-1, 1) + query_areas.view(1, -1) - iw * ih
    overlaps = iw * ih / ua
    return out_fn(overlaps)


def delete_all(demo_path, fmt="jpg"):
    """
    Delete all files with the given extension under `demo_path`.
    Useful for cleaning demo/visualization outputs between runs.

    Parameters
    ----------
    demo_path : str
        Directory to scan.
    fmt : str
        File extension (default: 'jpg').
    """

    import glob

    filelist = glob.glob(os.path.join(demo_path, "*." + fmt))
    if len(filelist) > 0:
        for f in filelist:
            os.remove(f)


def filter_tracklets(frame, tracks, look_back_size, batch_size, min_trklt_size):
    """
    Filter tracklets so that:
      (a) they appear somewhere in the current batch window, and
      (b) they have at least `min_trklt_size` observations (frames).
    Then crop each surviving tracklet to a look-back horizon.

    Parameters
    ----------
    frame : int
        Current (0-based) frame index.
    tracks : dict[int, dict[int, np.ndarray]]
        tracks[track_id][frame] = bbox (length-4 array).
    look_back_size : int
        Number of batches to look back.
    batch_size : int
        Number of frames per batch.
    min_trklt_size : int
        Minimum number of frames for a tracklet to be kept.

    Returns
    -------
    keep_tracks : dict[int, dict[int, np.ndarray]]
        Subset of `tracks` cropped to the look-back window.
    """

    # tracklets have items from the begining
    keep_tracks = {}
    # Current batch window: [frame - batch_size + 1, frame]
    batch_frames = range(frame - batch_size + 1, frame + 1)
    # Look-back window covering `look_back_size` batches
    look_back_frames = range(frame - look_back_size * batch_size + 1, frame + 1)

    for id, track in tracks.items():
        # Frames where this track_id appears
        set_trk_frames = set([fr for fr in track.keys()])  # set of frames in tracklet
        # Frames in the current batch
        set_cur_frames = set(batch_frames)  # set of frames of current batch
        if (
            len(set_trk_frames.intersection(set_cur_frames)) > 0
            and len(track.keys()) >= min_trklt_size
        ):
            # tracks key might be initiated with empty values: batch results will have only keys but no values
            look_back_batches = {
                k: v for k, v in track.items() if k in look_back_frames
            }

            # keep track as global key if it is available in look back frames otherwise return empty keep_tracks
            if len(look_back_batches.keys()) > 0:
                keep_tracks[id] = look_back_batches
    return keep_tracks


def plot_scanned_track(num_frames, tracks, im_path, output_dir, cam=None):
    """Plots a single image with the boxes from the last frame in `tracks`.

    Args:
        tracks (dict): The dictionary containing track boxes like tracks[track_id][frame] = bb.
        db (torch.utils.data.Dataset): (Unused here; kept for signature compatibility.)
        output_dir (String): Directory where to save the resulting image.

    Notes
    -----
    Visualization helper for manual inspection (does not affect tracking logic).
    """

    print("[*] Plotting scanned tracks at {}".format(osp.basename(im_path)))

    # Infinite color loop for track IDs
    cyl = cy("ec", colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    im_name = osp.basename(im_path)
    assert cam != None
    im_output = osp.join(output_dir, str(cam) + im_name)
    im = cv2.imread(im_path)
    # Convert BGR (OpenCV) to RGB (matplotlib)
    im = im[:, :, (2, 1, 0)]

    sizes = np.shape(im)
    height = float(sizes[0])
    width = float(sizes[1])

    # Create a full-bleed axes covering the image
    fig = plt.figure()
    fig.set_size_inches(width / 100, height / 100)
    ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
    ax.set_axis_off()
    fig.add_axes(ax)
    ax.imshow(im)

    # Draw the box for each track if it exists at the last frame (num_frames - 1)
    for j, t in tracks.items():
        # pdb.set_trace()
        if num_frames - 1 in t.keys():
            t_i = t[num_frames - 1]
            ax.add_patch(
                plt.Rectangle(
                    (t_i[0], t_i[1]),
                    t_i[2] - t_i[0],
                    t_i[3] - t_i[1],
                    fill=False,
                    edgecolor="green",
                    linewidth=2.0,
                    alpha=0.8,
                )
            )

            # Put the track id at the center of the box
            ax.annotate(
                j,
                (t_i[0] + (t_i[2] - t_i[0]) / 2.0, t_i[1] + (t_i[3] - t_i[1]) / 2.0),
                color="red",
                weight="bold",
                fontsize=20,
                ha="center",
                va="center",
            )

    plt.axis("off")
    # plt.tight_layout()
    # DefaultSize = plt.get_size_inches()
    # plt.set_figsize_inches((DefaultSize[0] // 2, DefaultSize[1] // 2))
    plt.draw()
    plt.savefig(im_output, dpi=100)
    plt.close()


def plot_sequence(tracks, db, output_dir):
    """Plot a whole sequence: every 30th frame gets an image with boxes overlaid.

    Args:
        tracks (dict): tracks[track_id][frame] = bb
        db (torch.utils.data.Dataset): The dataset with the images for the sequence.
        output_dir (String): Directory where to save the images.

    Notes
    -----
    Pure visualization helper for debugging/inspection.
    """

    print("[*] Plotting whole sequence to {}".format(output_dir))
    delete_all(demo_path=output_dir)
    if not osp.exists(output_dir):
        os.makedirs(output_dir)

    # Infinite color loop per track id
    cyl = cy("ec", colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    for i, v in enumerate(db):
        if i % 30 == 0:
            im_path = v["img_path"]
            im_name = osp.basename(im_path)
            im_output = osp.join(output_dir, im_name)
            im = cv2.imread(im_path)
            im = im[:, :, (2, 1, 0)]    # BGR->RGB

            sizes = np.shape(im)
            height = float(sizes[0])
            width = float(sizes[1])

            fig = plt.figure()
            fig.set_size_inches(width / 100, height / 100)
            ax = plt.Axes(fig, [0.0, 0.0, 1.0, 1.0])
            ax.set_axis_off()
            fig.add_axes(ax)
            ax.imshow(im)

            # Draw each track box if present at frame i
            for j, t in tracks.items():
                # pdb.set_trace()
                if i in t.keys():
                    t_i = t[i]
                    ax.add_patch(
                        plt.Rectangle(
                            (t_i[0], t_i[1]),
                            t_i[2] - t_i[0],
                            t_i[3] - t_i[1],
                            fill=False,
                            linewidth=2.0,
                            **styles[j]
                        )
                    )

                    ax.annotate(
                        j,
                        (
                            t_i[0] + (t_i[2] - t_i[0]) / 2.0,
                            t_i[1] + (t_i[3] - t_i[1]) / 2.0,
                        ),
                        color=styles[j]["ec"],
                        weight="bold",
                        fontsize=20,
                        ha="center",
                        va="center",
                    )

            plt.axis("off")
            # plt.tight_layout()
            # DefaultSize = plt.get_size_inches()
            # plt.set_figsize_inches((DefaultSize[0] // 2, DefaultSize[1] // 2))
            plt.draw()
            plt.savefig(im_output, dpi=100)
            plt.close()


def center_distances(inactive_pos, new_pos):
    """
    Compute pairwise Euclidean distances between **bottom-center** points of boxes.

    Why bottom-center?
    ------------------
    Using (x_center, y_bottom) approximates a person's/plant's footpoint or base,
    often more stable for association than the geometric center when height varies.

    Parameters
    ----------
    inactive_pos : torch.Tensor [N_inactive, 4]
        Boxes from inactive tracks [x1, y1, x2, y2].
    new_pos : torch.Tensor [N_new, 4]
        New detection boxes [x1, y1, x2, y2].

    Returns
    -------
    torch.Tensor [N_inactive, N_new]
        Pairwise Euclidean distances (in pixels).
    """

    # Compute (x_center, y_bottom) = (x1,y1) + ([w/2], [h]) = (x,y) + (dx,dy)
    inactive_cents = (
        inactive_pos[:, 0:2]
        + (inactive_pos[:, 2:4] - inactive_pos[:, 0:2])
        / torch.tensor([2.0, 1.0]).cuda()
    )
    new_cents = (
        new_pos[:, 0:2]
        + (new_pos[:, 2:4] - new_pos[:, 0:2]) / torch.tensor([2.0, 1.0]).cuda()
    )

    # torch.cdist computes all pairwise L2 distances
    return torch.cdist(inactive_cents, new_cents)


# def ReID_search_distances(new_pos, max_dist=300, max_patience=40):
#     """
#     Example idea: adapt search radius/patience as a function of distance from
#     the field-of-view center (smaller near edges, larger near center).
#     """
#
#     norm_factor = torch.tensor([1280, 800], dtype=torch.float).cuda()
#     fov_cent = torch.tensor([1280 // 2, 800 // 2], dtype=torch.float).cuda()
#     fov_cent = fov_cent.reshape(1, 2)

#     new_cents = new_pos[:, 0:2] + (new_pos[:, 2:4] - new_pos[:, 0:2]) / torch.tensor([2., 1.]).cuda()
#     cdist = torch.cdist(new_cents/norm_factor, fov_cent/norm_factor)
#     rads, pats = torch.exp(-2*cdist)*max_dist, torch.floor(torch.exp(-2*cdist)*max_patience)
#     return rads, pats


def plot_tracks(blobs, tracks, gt_tracks=None, output_dir=None, name=None):
    """
    Render two consecutive frames side-by-side with predicted tracks (and optional GT).

    Parameters
    ----------
    blobs : dict
        Must contain 'im_paths' (two image paths) and 'im_info' for scaling.
    tracks : torch.Tensor [T, 2, 4]
        Predicted boxes for two frames per track.
    gt_tracks : Optional[list[np.ndarray[4]]]
        Optional ground-truth boxes for overlay.
    output_dir : Optional[str]
        If provided, save the rendered image to disk.
    name : Optional[str]
        Custom output basename (without extension).

    Returns
    -------
    np.ndarray or None
        RGB image array if `output_dir` is None; otherwise None after saving.
    """

    # output_dir = get_output_dir("anchor_gt_demo")
    im_paths = blobs["im_paths"]
    if not name:
        im0_name = osp.basename(im_paths[0])
    else:
        im0_name = str(name) + ".jpg"
    im0 = cv2.imread(im_paths[0])
    im1 = cv2.imread(im_paths[1])
    im0 = im0[:, :, (2, 1, 0)]  # BGR->RGB
    im1 = im1[:, :, (2, 1, 0)]  # BGR->RGB

    # Scale factor to map back to original image coordinates
    im_scales = blobs["im_info"][0, 2]
    
    # Convert boxes to original pixel coordinates
    tracks = tracks.data.cpu().numpy() / im_scales
    num_tracks = tracks.shape[0]

    fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    ax[0].imshow(im0, aspect="equal")
    ax[1].imshow(im1, aspect="equal")

    # Infinite color loop across track index
    cyl = cy("ec", colors)
    loop_cy_iter = cyl()
    styles = defaultdict(lambda: next(loop_cy_iter))

    ax[0].set_title(("{} tracks").format(num_tracks), fontsize=14)

    # Draw each track in both frames
    for i, t in enumerate(tracks):
        t0 = t[0]
        t1 = t[1]
        ax[0].add_patch(
            plt.Rectangle(
                (t0[0], t0[1]),
                t0[2] - t0[0],
                t0[3] - t0[1],
                fill=False,
                linewidth=1.0,
                **styles[i]
            )
        )

        ax[1].add_patch(
            plt.Rectangle(
                (t1[0], t1[1]),
                t1[2] - t1[0],
                t1[3] - t1[1],
                fill=False,
                linewidth=1.0,
                **styles[i]
            )
        )

    # Optionally overlay ground truth
    if gt_tracks:
        for gt in gt_tracks:
            for i in range(2):
                ax[i].add_patch(
                    plt.Rectangle(
                        (gt[i][0], gt[i][1]),
                        gt[i][2] - gt[i][0],
                        gt[i][3] - gt[i][1],
                        fill=False,
                        edgecolor="blue",
                        linewidth=1.0,
                    )
                )

    plt.axis("off")
    plt.tight_layout()
    plt.draw()
    image = None
    if output_dir:
        im_output = osp.join(output_dir, im0_name)
        plt.savefig(im_output)
    else:
        image = np.fromstring(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


def interpolate(tracks):
    """
    Linearly interpolate per-track boxes across missing frames for smoothing/visualization.

    Parameters
    ----------
    tracks : dict[int, dict[int, np.ndarray[4]]]
        tracks[track_id][frame] = [x1,y1,x2,y2]

    Returns
    -------
    interpolated : dict[int, dict[int, np.ndarray[4]]]
        Each track gets boxes for all frames between min..max of its original keys.
    """

    interpolated = {}
    for i, track in tracks.items():
        interpolated[i] = {}
        frames = []
        x0 = []
        y0 = []
        x1 = []
        y1 = []

        # Gather frame indices and per-coordinate sequences
        for f, bb in track.items():
            frames.append(f)
            x0.append(bb[0])
            y0.append(bb[1])
            x1.append(bb[2])
            y1.append(bb[3])

        if len(frames) > 1:
            # Build 1D interpolants for each coordinate
            x0_inter = interp1d(frames, x0)
            y0_inter = interp1d(frames, y0)
            x1_inter = interp1d(frames, x1)
            y1_inter = interp1d(frames, y1)

            # Fill all integer frames between min and max
            for f in range(min(frames), max(frames) + 1):
                bb = np.array([x0_inter(f), y0_inter(f), x1_inter(f), y1_inter(f)])
                interpolated[i][f] = bb
        else:
            # Single frame -> copy as-is
            interpolated[i][frames[0]] = np.array([x0[0], y0[0], x1[0], y1[0]])

    return interpolated


def bbox_transform_inv(boxes, deltas):
    """
    Inverse bounding-box transform used by Faster R-CNN-style regressors.

    Given anchor/proposal boxes and deltas (tx,ty,tw,th),
    return [x1,y1,x2,y2] for each.

    Returns
    -------
    torch.Tensor [N, 4]
        Transformed boxes [x1,y1,x2,y2].
    """
    # If there are no boxes, return zeros matching the shape of deltas
    if len(boxes) == 0:
        return deltas.detach() * 0

    # Width/height and centers of anchors/proposals
    widths = boxes[:, 2] - boxes[:, 0] + 1.0
    heights = boxes[:, 3] - boxes[:, 1] + 1.0
    ctr_x = boxes[:, 0] + 0.5 * widths
    ctr_y = boxes[:, 1] + 0.5 * heights

    # Split deltas
    dx = deltas[:, 0::4]
    dy = deltas[:, 1::4]
    dw = deltas[:, 2::4]
    dh = deltas[:, 3::4]

    # Apply deltas: translate centers and scale widths/heights
    pred_ctr_x = dx * widths.unsqueeze(1) + ctr_x.unsqueeze(1)
    pred_ctr_y = dy * heights.unsqueeze(1) + ctr_y.unsqueeze(1)
    pred_w = torch.exp(dw) * widths.unsqueeze(1)
    pred_h = torch.exp(dh) * heights.unsqueeze(1)

    # Convert (cx,cy,w,h) -> (x1,y1,x2,y2)
    pred_boxes = torch.cat(
        [
            _.unsqueeze(2)
            for _ in [
                pred_ctr_x - 0.5 * pred_w,
                pred_ctr_y - 0.5 * pred_h,
                pred_ctr_x + 0.5 * pred_w,
                pred_ctr_y + 0.5 * pred_h,
            ]
        ],
        2,
    ).view(len(boxes), -1)
    return pred_boxes


def clip_boxes(boxes, im_shape):
    """
    Clip boxes to image boundaries (x ∈ [0, W-1], y ∈ [0, H-1]).

    Notes
    -----
    Kept to match existing call sites. For pure torchvision tensors,
    `torchvision.ops.clip_boxes_to_image` is usually preferred.
    """
    """
    Clip boxes to image boundaries.
    boxes must be tensor or Variable, im_shape can be anything but Variable
    """

    if not hasattr(boxes, "data"):
        boxes_ = boxes.numpy()

    # Reshape to [..., 4] and clamp each coordinate to image range
    boxes = boxes.view(boxes.size(0), -1, 4)
    boxes = torch.stack(
        [
            boxes[:, :, 0].clamp(0, im_shape[1] - 1),
            boxes[:, :, 1].clamp(0, im_shape[0] - 1),
            boxes[:, :, 2].clamp(0, im_shape[1] - 1),
            boxes[:, :, 3].clamp(0, im_shape[0] - 1),
        ],
        2,
    ).view(boxes.size(0), -1)

    return boxes


def get_center(pos):
    """
    Return the center (cx, cy) of a single box tensor [1,4].
    CUDA tensor is returned to align with downstream GPU ops.
    """
    
    x1 = pos[0, 0]
    y1 = pos[0, 1]
    x2 = pos[0, 2]
    y2 = pos[0, 3]
    return torch.Tensor([(x2 + x1) / 2, (y2 + y1) / 2]).cuda()


def get_width(pos):
    """Box width for a [1,4] tensor."""
    return pos[0, 2] - pos[0, 0]


def get_height(pos):
    """Box height for a [1,4] tensor."""
    return pos[0, 3] - pos[0, 1]


def make_pos(cx, cy, width, height):
    """
    Build a [1,4] (xyxy) box from (cx, cy, w, h).
    Handy for motion/extrapolation blocks that operate in center-size space.
    """

    return torch.Tensor(
        [[cx - width / 2, cy - height / 2, cx + width / 2, cy + height / 2]]
    ).cuda()


def warp_pos(pos, warp_matrix):
    """
    Apply a 2x3 affine (or 3x3 homography's upper 2x3) transform to a box
    by transforming its two opposite corners.

    Context
    -------
    Used when a global motion estimate between frames is available (ECC/LoFTR/SLAM),
    allowing a box from frame t to be mapped into frame t+1 coordinates.

    Parameters
    ----------
    pos : torch.Tensor [1,4]
        Source box (xyxy).
    warp_matrix : torch.Tensor [2x3] or [3x3]
        The linear transform (affine) or the top-left 2x3 of a homography.

    Returns
    -------
    torch.Tensor [1,4]
        Transformed box in destination coordinates.
    """

    # Homogeneous coordinates for the two corners
    p1 = torch.Tensor([pos[0, 0], pos[0, 1], 1]).view(3, 1)
    p2 = torch.Tensor([pos[0, 2], pos[0, 3], 1]).view(3, 1)

    # Apply transform
    p1_n = torch.mm(warp_matrix, p1).view(1, 2)
    p2_n = torch.mm(warp_matrix, p2).view(1, 2)

    # Reassemble xyxy on CUDA
    return torch.cat((p1_n, p2_n), 1).view(1, -1).cuda()


def get_mot_accum(results, seq):
    """
    Build a motmetrics `MOTAccumulator` by feeding per-frame GT and predictions.

    Procedure
    ---------
    - For each frame, compute an IoU-based distance matrix (via motmetrics'
      `iou_matrix` with `max_iou=0.5`).
    - Update the accumulator with GT ids, predicted ids, and the distance matrix.
    - Later, pass the accumulator(s) to `evaluate_mot_accums` to compute metrics.

    Parameters
    ----------
    results : dict[int, dict[int, np.ndarray(5,)]]
        results[track_id][frame_idx] = [x1,y1,x2,y2,score].
    seq : iterable[dict]
        Each element is a frame dict with 'gt' mapping {gt_id: [x1,y1,x2,y2]}.

    Returns
    -------
    motmetrics.MOTAccumulator
        Accumulator filled across the sequence.
    """

    mot_accum = mm.MOTAccumulator(auto_id=True)

    for i, data in enumerate(seq):
        gt = data["gt"]
        gt_ids = []
        if gt:
            # Collect GT boxes in the order of their ids
            gt_boxes = []
            for gt_id, box in gt.items():
                gt_ids.append(gt_id)
                gt_boxes.append(box)

            gt_boxes = np.stack(gt_boxes, axis=0)
            # Convert xyxy -> xywh as required by motmetrics' IoU util
            gt_boxes = np.stack(
                (
                    gt_boxes[:, 0],
                    gt_boxes[:, 1],
                    gt_boxes[:, 2] - gt_boxes[:, 0],
                    gt_boxes[:, 3] - gt_boxes[:, 1],
                ),
                axis=1,
            )
        else:
            gt_boxes = np.array([])

        # Collect predictions present at frame i
        track_ids = []
        track_boxes = []
        for track_id, frames in results.items():
            if i in frames:
                track_ids.append(track_id)
                # frames[i] = [x1,y1,x2,y2,score] -> take only the first 4 values
                track_boxes.append(frames[i][:4])

        if track_ids:
            track_boxes = np.stack(track_boxes, axis=0)
            # Convert xyxy -> xywh like GT
            track_boxes = np.stack(
                (
                    track_boxes[:, 0],
                    track_boxes[:, 1],
                    track_boxes[:, 2] - track_boxes[:, 0],
                    track_boxes[:, 3] - track_boxes[:, 1],
                ),
                axis=1,
            )
        else:
            track_boxes = np.array([])

        # IoU-based distance matrix (motmetrics handles 1 - IoU internally via max_iou)
        distance = mm.distances.iou_matrix(gt_boxes, track_boxes, max_iou=0.5)

        # Update accumulator with id lists and the distance matrix for frame i
        mot_accum.update(gt_ids, track_ids, distance)

    return mot_accum


def evaluate_mot_accums(accums, names, generate_overall=False):
    """
    Compute and print standard MOT metrics (MOTA, IDF1, etc.) for multiple accumulators.

    Parameters
    ----------
    accums : list[motmetrics.MOTAccumulator]
        Accumulators previously filled by `get_mot_accum`.
    names : list[str]
        Display names (e.g., sequence names) for each accumulator.
    generate_overall : bool
        If True, adds an 'OVERALL' aggregate row.

    Side effects
    ------------
    Prints a formatted summary table to stdout.
    """
    
    mh = mm.metrics.create()
    summary = mh.compute_many(
        accums,
        metrics=mm.metrics.motchallenge_metrics,
        names=names,
        generate_overall=generate_overall,
    )

    str_summary = mm.io.render_summary(
        summary,
        formatters=mh.formatters,
        namemap=mm.io.motchallenge_metric_names,
    )
    print(str_summary)
