import os
import cv2
import yaml
import time
import tqdm
import torch
import numpy as np
import matplotlib.pyplot as plt

import kornia
import kornia.feature as KF
from ultralytics import YOLO

from ..models.frcnn_fpn import FRCNN_FPN

# from .tracktor.tracker_clean import Tracker
from .tracktor.slam import SLAM
from .tracktor.reid.resnet import resnet50
from .tracktor import tracker_ag as ag
from .tracktor import tracker_agt as agt
from .tracktor import tracker_orig as orig
from .tracktor import tracker_clean as clean

from config import tracker_ag_config as cfg_ag
from config import tracker_agt_config as cfg_agt
from config import tracker_orig_config as cfg_orig
from config import tracker_clean_config as cfg_clean

np.set_printoptions(precision=3, suppress=True)


def byte_track(cfg, datasets):
    """
    Run ByteTrack with YOLOv8 on each dataset sequence and save MOT-format results.

    Parameters
    ----------
    cfg : dict
        Experiment config; must include:
        - "detector": must be "yolov8" for this function,
        - "yolo_maker": YOLO model checkpoint path/name for Ultralytics,
        - "outdir": base directory to write results under `trackers/bytetrack/`.
    datasets : list
        List of dataset objects; each is expected to expose:
        - `.imgdir` : path to current sequence image folder,
        - `.files`  : ordered list of frame image file paths.

    Output
    ------
    Writes `<outdir>/trackers/bytetrack/<sequence>.txt` with lines:
        frame, id, x, y, w, h, 1, 1, 1, 1
    where (x, y, w, h) are left/top/width/height in pixels, `frame` is 1-based.
    """
    if cfg["detector"] == "yolov8":
        # Construct YOLOv8 model from a checkpoint or model name.
        model = YOLO(cfg["yolo_maker"])
    else:
        # Guardrail: ByteTrack path only supports YOLOv8 detector here.
        raise ValueError("Invalid detector for ByteTrack tracker")
    
    # Iterate over each dataset/sequence
    for dataset in datasets:
        print("\n" * 2 + "dataset:", dataset.imgdir)

        # Ensure output directory exists
        os.makedirs(cfg["outdir"] + "trackers/bytetrack", exist_ok=True)

        # Compose result file name from the parent of imgdir (sequence identifier)
        out_file = (
            cfg["outdir"]
            + "trackers/bytetrack/"
            + dataset.imgdir.split("/")[-3]
            + ".txt"
        )
        file = open(out_file, "w")

        # Process each frame path with YOLOv8 tracking
        for i, path in tqdm.tqdm(enumerate(dataset.files), desc="Tracking"):
            results = model.track(
                source=path,    # image path (single frame)
                tracker="config/apples_bytetrack.yaml", # ByteTrack cfg for Ultralytics
                verbose=False,  # suppress console noise
                persist=True,   # keep track state across frames
            )

            # Skip frames with no tracks
            if results[0].boxes.id is None:
                continue
            
            # Get (x,y,w,h) in XYWH center-based coords from Ultralytics
            boxes = results[0].boxes.xywh.cpu()
            # Track IDs as Python ints
            track_ids = results[0].boxes.id.int().cpu().tolist()

            # Convert to MOT format: left, top, width, height from center XYWH
            for box, track_id in zip(boxes, track_ids):
                left = box[0] - box[2] / 2
                top = box[1] - box[3] / 2
                file.write(
                    "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n"
                    % (i + 1, track_id, left, top, box[2], box[3], 1, 1, 1, 1)
                )
        file.close()


def track(cfg, loader, name):
    """
    Run Tracktor-style tracking for a given sequence and write MOT-format results.

    Parameters
    ----------
    cfg : dict
        Experiment/tracking config; keys used include:
        - "tracker": one of {"clean","orig","agt","ag"},
        - "device": e.g., "cuda:0",
        - "backbone": detector backbone label (FRCNN_FPN),
        - "checkpoint": path to Faster R-CNN weights,
        - "outdir": base output directory.
    loader : torch.utils.data.DataLoader
        Dataloader that yields `(img, target)` where:
        - `img` is a batch-like structure; `img[0]` is the image tensor,
        - `target` is a dict with `"img"`: original image tensor (C,H,W) in [0,1].
    name : str
        Sequence name; used in output file naming.

    Output
    ------
    Writes: `<outdir>/trackers/{tracker}/{name}.txt` with MOT lines:
        frame, id, x, y, w, h, conf, 1, 1, 1
    Also writes timing plot under `<outdir>/trackers/{tracker}/times/{name}.jpg`
    and prints min/mean/max iteration times.
    """
    # Select tracker-specific configuration
    if cfg["tracker"] == "clean":
        tracker_cfg = cfg_clean.get_config()
    if cfg["tracker"] == "orig":
        tracker_cfg = cfg_orig.get_config()
    if cfg["tracker"] == "agt":
        tracker_cfg = cfg_agt.get_config()
    if cfg["tracker"] == "ag":
        tracker_cfg = cfg_ag.get_config()

    # Select compute device (e.g., "cuda:0")
    device = torch.device(cfg["device"])

    # ------------------- Detector setup -------------------
    print("Detector: setting up...")
    detector = FRCNN_FPN(cfg["backbone"])   # instantiate Faster R-CNN + FPN
    print('Detector: loading checkpoint: "%s"' % cfg["checkpoint"])
    detector.load_state_dict(torch.load(cfg["checkpoint"])) # restore weights
    detector.to(device)
    detector.eval()
    print("Detector ready!")

    # ------------------- ReID setup (if required) -------------------
    if cfg["tracker"] in ["orig", "ag"]:        
        print("ReID network: setting up...")        
        # Load ReID YAML to configure the ResNet-50 backbone
        with open(tracker_cfg.REID_CONFIG) as file:
            reid_config = yaml.load(file, Loader=yaml.FullLoader)
        reid_network = resnet50(pretrained=False, **reid_config["reid"]["cnn"])
        print('ReID network: loading checkpoint: "%s"' % tracker_cfg.REID_WEIGHTS)
        reid_network.load_state_dict(torch.load(tracker_cfg.REID_WEIGHTS))
        reid_network.to(device)
        reid_network.eval()
        print("ReID network ready!")

    # ------------------- Tracker choice -------------------
    if cfg["tracker"] == "clean":
        tracker = clean.Tracker(detector, tracker_cfg)
    if cfg["tracker"] == "orig":
        tracker = orig.Tracker(detector, reid_network, tracker_cfg)
    if cfg["tracker"] == "agt":
        tracker = agt.Tracker(detector, cfg, tracker_cfg)
    if cfg["tracker"] == "ag":
        tracker = ag.Tracker(detector, reid_network, tracker_cfg)

    # Prepare outputs: MOT text, render and timing directories
    outfile = open(cfg["outdir"] + "trackers/%s/%s.txt" % (cfg["tracker"], name), "w")
    os.makedirs((cfg["outdir"] + "trackers/%s/render/" % cfg["tracker"]), exist_ok=True)
    os.makedirs((cfg["outdir"] + "trackers/%s/times/" % cfg["tracker"]), exist_ok=True)
    
    # Disable grads for pure inference; measure per-iteration times
    with torch.no_grad():
        t0, t1 = time.time(), time.time()
        times = []

        # Iterate over frames
        for i, (img, target) in tqdm.tqdm(enumerate(loader), desc="Tracking"):
            # Take the first (and only) image from the loader batch and add batch dim
            imgT = img[0][None, :, :, :].to(device)

            # Recover original float image for optional rendering/IO
            img = target["img"][0].cpu().numpy()
            B, G, R = img[0, :, :], img[1, :, :], img[2, :, :]
            img = np.stack([B, G, R], 2)

            # Run detector in a "batched" helper that exposes features and sizes
            dets, features, sizes = detector.detect_batches(imgT)

            # Advance tracker state with current frame's inputs and detections
            tracker.step(
                {"img": imgT, "net_in_sizes": sizes[0], "features": features}, dets[0]
            )

            # Export current-frame tracks to MOT format (if any end at this frame)
            results = tracker.get_results()
            for track in results.keys():
                for frame in results[track].keys():
                    if frame == i:
                        x1, y1, x2, y2, conf = results[track][frame]
                        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                        outfile.write(
                            "%d,%d,%d,%d,%d,%d,%d,%d,%d,%d\n"
                            % (i + 1, track, x1, y1, x2 - x1, y2 - y1, conf, 1, 1, 1)
                        )
            #             cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 1), 3)
            #             cv2.putText(img, str(track), (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, (1, 0, 0), 3)
            # cv2.imshow('result', np.transpose(img, (1, 0, 2)))
            # cv2.imshow('result', img)
            # if i%1 == 0: cv2.imwrite(cfg['outdir'] + 'track/render/%d.jpg' % (i + 1), (img * 255).astype(np.uint8))
            # cv2.waitKey(0)

            # (Optional drawing & preview code kept commented to avoid I/O overhead.)
            # t0/t1 timing to compute per-iteration latency
            t1 = time.time()
            times.append(t1 - t0)
            t0 = t1
    
    # Close outputs and plot iteration-time statistics (skip first two warm-up steps)
    outfile.close()
    times = np.array(times[2:])
    plt.plot(times)
    plt.title("%3.4f, %3.4f, %3.4f" % (times.min(), times.mean(), times.max()))
    plt.savefig(cfg["outdir"] + "trackers/%s/times/%s.jpg" % (cfg["tracker"], name))
    plt.clf()
    print("Times:", times.min(), times.mean(), times.max())


def slam(cfg, loader, name):
    """
    Produce a homography-based stitched mosaic and trajectory plots for a sequence.

    Parameters
    ----------
    cfg : dict
        Config with "outdir" base directory and other SLAM-related parameters.
    loader : torch.utils.data.DataLoader
        Iterable over frames of the sequence; yields `(imgT, target)` pairs where:
        - `imgT` is an image-like tensor used by the SLAM pipeline,
        - `target["img"][0]` stores the original float image in (C,H,W), [0,1].
    name : str
        Sequence name; used in file naming.

    Output
    ------
    - Saves per-frame stitched images under `<outdir>/slam/render/<name>/`.
    - Saves timing plot under `<outdir>/slam/times/<name>.jpg`.
    - Saves overall stitched mosaic image `<outdir>/slam/<name>.jpg`.
    - Saves Matplotlib trajectory overlays (JPG/EPS) in `<outdir>/slam/`.
    """
    # Ensure output directories exist for render frames and timing plots
    os.makedirs(cfg["outdir"] + "slam/render/", exist_ok=True)
    os.makedirs(cfg["outdir"] + "slam/times", exist_ok=True)

    # Load default parameters from AGT tracker config (reduction, RANSAC threshold)
    agt_cfg = cfg_agt.get_config()

    # Construct SLAM engine; operates on image tensors and sizes each frame
    slam = SLAM(cfg, agt_cfg.REDUCTION_FACTOR, agt_cfg.RANSAC_THRES)
    
    # camera position centers extracted periodically
    cam_pos = []    
    
    with torch.no_grad():
        t0, t1 = time.time(), time.time()
        times = []
        # Track the stitched canvas bounds (top-left & bottom-right in mosaic space)
        tmix, tmiy, tmax, tmay = 0, 0, 0, 0

        # Iterate through frames
        for i, (imgT, target) in tqdm.tqdm(enumerate(loader)):
            # Recover original float image for stitching/warping (HWC with BGR order)
            img = target["img"][0].cpu().numpy()
            B, G, R = img[0, :, :], img[1, :, :], img[2, :, :]
            img = np.stack([B, G, R], 2)

            if i == 0:
                # Initialize mosaic with the first frame and track extents
                im0 = img
                tmix, tmiy, tmax, tmay = 0, 0, img.shape[1], img.shape[0]

            if i == 301:
                # Hard cut-off for sequence length (dataset-specific comment hints)
                break  # straight4:7:301, B&F1:0:361, O&I2:3:851

            # Current tensor spatial size (for warping math)
            h, w = imgT.shape[-2], imgT.shape[-1]

            # Run SLAM for this frame; updates internal homography H
            slam.run_frame(imgT, h, w, i)
            H0 = slam.H     # 3x3 homography for current frame

            # Sample camera center every 10 frames for coarse path plotting
            if i % 10 == 0:
                cam_pos.append(
                    (
                        H0 @ np.array([[img.shape[1] / 2, img.shape[0] / 2, 1]]).T
                    ).flatten()[:2]
                )

            # ---------- Rendering and Warping ----------
            # Build corners in image coords and warp them using H0
            corners = np.array([[0, w, 0, w], [0, 0, h, h], [1, 1, 1, 1]])
            warpcorners = H0 @ corners
            wc = np.round(warpcorners[:2, :] / warpcorners[2, :]).astype(int)

            # Previous and current bounding boxes in mosaic coordinates
            pmix, pmiy, pmax, pmay = tmix, tmiy, tmax, tmay
            cmix, cmiy, cmax, cmay = (
                wc[0, :].min(),
                wc[1, :].min(),
                wc[0, :].max(),
                wc[1, :].max(),
            )

            # Update total mosaic extents to include the new warped frame
            tmix, tmiy, tmax, tmay = (
                min(cmix, tmix),
                min(cmiy, tmiy),
                max(cmax, tmax),
                max(cmay, tmay),
            )

            # Translate transform to place current frame at (0,0) of its local canvas
            Haux = np.array([[1.0, 0.0, -cmix], [0.0, 1.0, -cmiy], [0, 0, 1]])

            # Local canvas sizes
            wt, ht = cmax - cmix, cmay - cmiy      # current warped frame canvas
            w0, h0 = tmax - tmix, tmay - tmiy      # global mosaic canvas so far

            # Warp current frame into its local canvas with nearest-neighbor borders
            frt = cv2.warpPerspective(
                img,
                Haux @ H0,
                (wt, ht),
                borderMode=cv2.BORDER_CONSTANT,
                flags=cv2.INTER_NEAREST,
                borderValue=(1, 1, 1),
            )

            # Warp a 1-mask alongside for alpha blending into the global mosaic
            frt_mask = cv2.warpPerspective(
                np.ones_like(img),
                Haux @ H0,
                (wt, ht),
                borderMode=cv2.BORDER_CONSTANT,
                flags=cv2.INTER_NEAREST,
                borderValue=(0, 0, 0),
            )

            # Allocate/extend the global mosaic canvas (initialized to ones)
            full = np.ones((h0, w0, 3))

            # Paste the previous mosaic into the new (potentially larger) canvas
            full[
                pmiy - tmiy : pmiy - tmiy + im0.shape[0],
                pmix - tmix : pmix - tmix + im0.shape[1],
            ] = im0

            # Alpha-like composite: where mask==0 keep existing pixels, else use warped
            frt = (frt_mask == 0) * (
                full[cmiy - tmiy : cmiy - tmiy + ht, cmix - tmix : cmix - tmix + wt]
            ) + (frt_mask == 1) * frt

            # Write the blended patch back to the mosaic
            full[cmiy - tmiy : cmiy - tmiy + ht, cmix - tmix : cmix - tmix + wt] = frt
            im0 = full  # update mosaic

            # ---------- Saving per-frame renders ----------
            imsv = (im0 * 255).astype(np.uint8)     # scale to [0,255] for image IO
            hsv, wsv, _ = imsv.shape

            # Resize for consistent visualization footprint (max dimension 2000 px)
            imsv = cv2.resize(
                imsv,
                (wsv * 2000 // max(hsv, wsv), hsv * 2000 // max(hsv, wsv)),
                interpolation=cv2.INTER_CUBIC,
            )
            cv2.imwrite(
                cfg["outdir"] + "slam/render/%s/%s_%04d.jpg" % (name, name, i), imsv
            )
            
            # ---------- Timing ----------
            t1 = time.time()
            times.append(t1 - t0)
            t0 = t1

        # Normalize name for file system and save final mosaic snapshot
        name = name.replace("&", "n")
        cv2.imwrite(cfg["outdir"] + "slam/%s.jpg" % name, imsv)

        # ---------- Matplotlib path plot (coordinates) ----------
        plt.figure(figsize=(6, 1.5))
        x = np.array(cam_pos)[:, 0]
        y = np.array(cam_pos)[:, 1]
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u / 2
        pos_y = y[:-1] + v / 2
        plt.plot(y, x, "b")
        plt.quiver(pos_y, pos_x, v, u, color="b", pivot="mid", width=2)
        # Fix aspect ratio
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Path $\mathbf{x}_t$", fontsize=8)
        plt.xlabel("y [pixels]", fontsize=8)
        plt.ylabel("x [pixels]", fontsize=8)
        # Guardrails for y-limits to keep axes readable across sequences
        plt.ylim([min(-50, min(x)), max(550, max(x))])
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)        
        # plt.tight_layout()
        # plt.savefig(cfg['outdir'] + 'slam/%s_path.jpg' % name, dpi=300)
        # plt.savefig(cfg['outdir'] + 'slam/%s_path.eps' % name, dpi=300)

        # (Optional tight_layout)
        # Save timing plot
        times = np.array(times[2:]) # drop first two timings (warm-up)
        plt.figure()
        plt.plot(times)
        plt.savefig(cfg["outdir"] + "slam/times/%s.jpg" % name)
        plt.clf()

        # ---------- Draw trajectory with OpenCV on the mosaic ----------
        imbk = imsv.copy()        
        hsv, wsv, _ = imsv.shape

        # Map cam_pos coordinates into current mosaic image coordinates
        cam_pos = np.array(cam_pos)
        cam_pos[:, 0] = cam_pos[:, 0] - tmix
        cam_pos[:, 1] = cam_pos[:, 1] - tmiy
        cam_pos[:, 0] = cam_pos[:, 0] * wsv / (tmax - tmix)
        cam_pos[:, 1] = cam_pos[:, 1] * hsv / (tmay - tmiy)
        cam_pos = cam_pos.astype(int)

        # Draw arrows between consecutive camera centers
        for i in range(len(cam_pos) - 1):
            cv2.arrowedLine(
                imsv,
                tuple(cam_pos[i]),
                tuple(cam_pos[i + 1]),
                (0, 0, 255),
                thickness=4,
                tipLength=0.2,
            )
        
        # (An alternative save is commented out.)
        # cv2.imwrite(cfg['outdir'] + 'slam/%s_image_path_opencv.jpg' % name, imsv)

        # Draw the trajectory on imsv using MATPLOTLIB
        # imsv = imbk.copy()

        # ---------- Matplotlib overlay on the rotated mosaic ----------
        # Rotate for display and convert BGR -> RGB for Matplotlib
        imsv = cv2.rotate(imsv, cv2.ROTATE_90_COUNTERCLOCKWISE)
        imsv = cv2.cvtColor(imsv, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 1.5))  # (6, 1), (6, 1.5), (6, 3)
        plt.imshow(imsv)
        # plt.plot(cam_pos[:, 1], wsv - cam_pos[:, 0], 'k', linewidth=0.8)

        # Prepare quiver arrows (normalized by step length if desired)
        x = cam_pos[:, 1]
        y = wsv - cam_pos[:, 0]
        u = np.diff(x)
        v = np.diff(y)
        pos_x = x[:-1] + u / 2
        pos_y = y[:-1] + v / 2
        norm = np.sqrt(u**2 + v**2)        
        # plt.quiver(pos_x, pos_y, u/norm, -v/norm, color='r', width=0.004)
        # Fix aspect ratio
        plt.gca().set_aspect("equal", adjustable="box")
        plt.title("Path $\mathbf{x}_t$", fontsize=8)
        plt.xlabel("y [pixels]", fontsize=8)
        plt.ylabel("x [pixels]", fontsize=8)
        plt.xticks(fontsize=8)
        plt.yticks(fontsize=8)
        # Save both raster and vector versions
        plt.savefig(cfg["outdir"] + "slam/%s_image_path_matplotlib.jpg" % name, dpi=300)
        plt.savefig(cfg["outdir"] + "slam/%s_image_path_matplotlib.eps" % name, dpi=300)
        plt.clf()
        plt.close()
