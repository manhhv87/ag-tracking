import os
import cv2
import tqdm
import numpy as np


def play(cfg, dataset, name):
    """
    Render detections/tracks frame-by-frame, preview them live, and export both images and an MP4.

    Pipeline:
      1) Create output folders under cfg["outdir"] for per-frame images and the final video.
      2) Open a cv2.VideoWriter (MP4v, 10 FPS) with a fixed frame size of (810, 1080).
      3) Iterate over `dataset` (yields (imgs, target)). For each element:
         - `target["img"][0]`   : image tensor (C, H, W) with values in [0..1] (by context).
         - `target["ids"][0]`   : 1D array of object IDs (one per bounding box).
         - `target["boxes"][0]` : Nx4 array of pixel boxes (x1, y1, x2, y2).
      4) Convert the tensor image to a NumPy array and rearrange channels to B,G,R for OpenCV.
      5) Draw bounding boxes and IDs on the frame, show it via `cv2.imshow`, and:
         - write the frame to the MP4,
         - save the frame as a .jpg under `play/image/{name}/`.
      6) After all frames, release the video writer.

    Args:
        cfg (dict): Output configuration; must contain:
                    - "outdir" (str): base directory to write `play/image/` and `play/video/`.
        dataset (Iterable): Generator/DataLoader yielding `(imgs, target)` per step.
                            Only `target` is used here, with keys:
                            - target["img"]   : tensor of shape (B, C, H, W); here B=1 is assumed.
                            - target["ids"]   : tensor of shape (B, N) with integer IDs.
                            - target["boxes"] : tensor of shape (B, N, 4) with (x1,y1,x2,y2).
        name (str): Session name used for file/folder names (e.g., `{name}.mp4`).

    Notes:
        - The video size is fixed at (810, 1080). If the incoming image size differs,
          OpenCV will still write, but content may be cropped or stretched upstream
          (no resize occurs here).
        - `cv2.imshow` requires a GUI/display. On headless servers, showing frames
          may fail; file writing still proceeds.
        - Drawing colors are given as very small tuples (0/1) because the image is
          multiplied by 255 before saving; OpenCV will cast to uint8.
        - `cv2.waitKey(1)` is used to keep the UI responsive; uncomment the
          `# if i == 50: break` snippet for quick early stopping during debugging.
    """    
    os.makedirs(cfg["outdir"] + "play/image/" + name, exist_ok=True)    # Ensure a destination folder exists for frame images    
    os.makedirs(cfg["outdir"] + "play/video", exist_ok=True)            # Ensure a destination folder exists for the output MP4 video

    # Configure codec and initialize the VideoWriter (MP4v, 10 FPS, frame size (810, 1080))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video = cv2.VideoWriter(
        cfg["outdir"] + "play/video/%s.mp4" % (name), fourcc, 10, (810, 1080)
    )
    
    # Iterate over the dataset with a progress bar
    for i, (imgs, target) in tqdm.tqdm(enumerate(dataset), desc="Playing"):
        # Optional debug: print shapes of image/boxes
        # print(name, 'frame', i, target['img'].shape, target['boxes'].shape)

        # Grab image from batch index 0 and convert to NumPy. Expect shape (C, H, W), C=3
        img = target["img"][0].numpy()
        # Object IDs (batch 0); cast to int for drawing/labeling
        ids = target["ids"][0].numpy().astype(int)
        # Bounding boxes (batch 0); cast to int for pixel indexing
        anns = target["boxes"][0].numpy().astype(int)

        # Split channels in B, G, R order to match OpenCV conventions
        B, G, R = img[0, :, :], img[1, :, :], img[2, :, :]
        # Stack back to an H x W x 3 image
        img = np.stack([B, G, R], 2)

        # Draw each bbox and its corresponding ID
        for id, bb in zip(ids, anns):
            x1, y1, x2, y2 = bb
            # Rectangle in (roughly) red after scaling (see the 255 multiplication below)
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 1), 2)
            # Put the track/detection ID near the bottom-left of the bbox; scale 1.5, thickness 2
            cv2.putText(
                img, str(id), (x1, y2 - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (1, 0, 0), 2
            )

        # Show the current frame (window named "video"); requires a GUI environment
        cv2.imshow("video", img)

        # Write one frame to the video file (convert [0..1] float to uint8 via *255)
        video.write((img * 255).astype(np.uint8))

        # Save the current frame as a JPG with zero-padded index
        cv2.imwrite(
            cfg["outdir"] + "play/image/%s/%06d.jpg" % (name, i),
            (img * 255).astype(np.uint8),
        )

        # Keep the UI responsive; 1 ms delay
        cv2.waitKey(1)
        
        # Early stop for quick inspection (currently commented)
        # if i == 50: break
    
    # Finalize the video file
    print("Saving video...")
    video.release()
    print("Video saved.")
