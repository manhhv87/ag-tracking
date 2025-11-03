import os
import numpy as np
from PIL import Image
from pycocotools import mask as mask_utils


def get_mot_gt(file):
    """
    Parse a MOTChallenge-style ground-truth file into a list of integer fields.

    Expected input format per line (CSV, no header):
        frame_id, track_id, x, y, w, h, ...

    The function:
      1) Reads all lines.
      2) Splits each line on commas.
      3) Casts the first six fields to int.
      4) Converts (x, y, w, h) into the [xmin, ymin, xmax, ymax] form by computing:
             xmax = x + w
             ymax = y + h
      5) Appends [frame, id, xmin, ymin, xmax, ymax] to the result list.

    Returns:
        List[List[int]]: Each item is [frame, id, xmin, ymin, xmax, ymax].

    Notes:
        - Only the first six CSV fields are used; any trailing fields are ignored.
        - This utility helps downstream tools that prefer corner coordinates.
        - It assumes all numeric tokens are valid integers (no error handling).
    """
    file = open(file, "r")
    lines = list(file.readlines())
    linesint = []
    for line in lines:
        fields = line[:-1].split(",")  # strip trailing newline then split
        frame = int(fields[0])
        id = int(fields[1])
        x = int(fields[2])
        y = int(fields[3])
        w = int(fields[4])
        h = int(fields[5])
        # Convert width/height to xmax/ymax; MOT stores top-left+size
        linesint.append([frame, id, x, y, x + w, y + h])
    return linesint


def encode_mask_to_RLE(binary_mask):
    """
    Encode a binary mask (H x W, {0,1}) to COCO's compressed RLE string.

    Args:
        binary_mask (np.ndarray): 2D uint8 or bool array, Fortran-contiguous preferred.

    Returns:
        str: COCO RLE 'counts' string (ascii).

    Implementation details:
        - COCO's pycocotools expects Fortran order; np.asfortranarray ensures that.
        - 'mask_utils' here refers to 'from pycocotools import mask as mask_utils'.
          This import is not present in this file; it must exist in the environment.

    Caveats:
        - This function is not used by default below (calls are commented out),
          but is kept for potential MOTS/segmentation export.
    """
    fortran_binary_mask = np.asfortranarray(binary_mask)
    rle = mask_utils.encode(fortran_binary_mask)
    return rle["counts"].decode("ascii")


def create_mot_n_coco_for_sequence(params):
    """
    Convert one AppleMOTS-style sequence into:
        (a) MOTChallenge 'gt.txt' for TrackEval, and
        (b) Ultralytics/YOLO-style label files (one .txt per image).

    It also symlinks/copies images into MOT and COCO-like directory layouts
    and writes a 'seqinfo.ini' describing the sequence.

    Args:
        params (dict): Required keys:
            - 'seq'  (str): Sequence name, e.g., '0001'.
            - 'set'  (str): Split name, e.g., 'train'/'val'/'test'.
            - 'input_dir' (str): Root folder containing AppleMOTS-style data with:
                  {input_dir}/{set}/images/{seq}/   (RGB frames, .jpg/.png)
                  {input_dir}/{set}/instances/{seq}/ (instance masks aligned to frames)
            - 'mot_save_dir' (str): Output root for MOT-style export. For each seq:
                  {mot_save_dir}/{seq}/gt/  (will contain gt.txt)
                  {mot_save_dir}/{seq}/img -> symlink to source images
            - 'coco_save_dir' (str): Output root for YOLO/Ultralytics export:
                  {coco_save_dir}/{set}/images/ (symlinks to source images)
                  {coco_save_dir}/{set}/labels/ (YOLO .txt labels)

    Returns:
        list: [frames, track_id, mask]
              - frames: number of frames processed (counted from instance files)
              - track_id: max track id modulo 1000 (as used in this conversion)
              - mask: total number of object masks encountered across frames

    Key behaviors/assumptions:
        - Only files ending in '.jpg' or '.png' are treated as frames.
        - Each mask image encodes instance ids as unique integer values; 0 is background.
        - Bounding boxes are computed from mask pixels (xmin/xmax/ymin/ymax).
        - YOLO labels are normalized using fixed image size (W=1296, H=972),
          which matches AppleMOTS resolution described in the paper’s dataset section.
        - Class id '47' is hard-coded for apples in the YOLO labels.
        - Image files are linked via symlinks (no deduplication if links exist).

    Potential pitfalls (left unchanged by design):
        - Existing symlinks: os.symlink will raise if the link already exists.
        - Box tightness: using min/max pixel positions yields inclusive bounds;
          width/height are computed as xmax - xmin and ymax - ymin (no +1).
        - 'seqinfo.ini' path uses string slicing 'mot_gt_dir[:-2]' to strip 'gt',
          which may yield a malformed path if directory names change.
        - The optional COCO RLE export is commented out; enabling it requires
          pycocotools and the 'mask_utils' alias.

    Relation to the paper:
        - The conversion mirrors the AppleMOTS→AppleMOT step (masks→boxes)
          used to evaluate box-based tracking (see dataset details in the paper).
    """
    print("Processing sequence %s" % params["seq"])

    # MOTS data directories (input, AppleMOTS layout)
    mots_images_dir = "%s/%s/images/%s" % (
        params["input_dir"],
        params["set"],
        params["seq"],
    )
    mots_instances_dir = "%s/%s/instances/%s" % (
        params["input_dir"],
        params["set"],
        params["seq"],
    )
    
    # MOT output data directories
    mot_gt_dir = "%s/%s/gt" % (params["mot_save_dir"], params["seq"])
    mot_img_dir = "%s/%s/img" % (params["mot_save_dir"], params["seq"])

    # Create MOT outputdirectories and link image data
    os.makedirs(mot_gt_dir, exist_ok=True)
    os.symlink(os.getcwd() + "/" + mots_images_dir, mot_img_dir)

    # COCO output data directories
    coco_labels_dir = "%s/%s/labels" % (params["coco_save_dir"], params["set"])
    coco_images_dir = "%s/%s/images" % (params["coco_save_dir"], params["set"])

    # Create COCO output directories
    os.makedirs(coco_labels_dir, exist_ok=True)
    os.makedirs(coco_images_dir, exist_ok=True)

    # Initialize MOT(S) counters
    #   mask  -> total number of object instances seen
    #   track -> max raw instance id encountered (to derive max track id later)
    #   frames-> number of frames processed
    mask, track, frames = 0, 0, 0

    # Open the MOT ground-truth file for this sequence
    with open("%s/gt.txt" % mot_gt_dir, "w") as gt_mot:
        # Iterate frames in instance-mask directory (sorted for temporal order)
        for filename in sorted(os.listdir(mots_instances_dir)):
            image_mots_path = "%s/%s" % (mots_images_dir, filename)
            instance_mots_path = "%s/%s" % (mots_instances_dir, filename)

            # Only process typical image files; skip anything else
            if filename.endswith(".jpg") or filename.endswith(".png"):
                frames += 1

                # Skip if corresponding files are missing
                if not os.path.isfile(image_mots_path):
                    continue
                if not os.path.isfile(instance_mots_path):
                    continue

                # Determine output (linked) image path and YOLO label path
                name, ext = filename.split(".")
                image_coco_path = "%s/%s%s.%s" % (
                    coco_images_dir,
                    params["seq"],
                    name,
                    ext,
                )
                label_coco_path = "%s/%s%s.%s" % (
                    coco_labels_dir,
                    params["seq"],
                    name,
                    "txt",
                )

                # Link COCO/YOLO image to the original AppleMOTS image
                os.symlink(os.getcwd() + "/" + image_mots_path, image_coco_path)
                
                # Create YOLO label file for this frame
                with open(label_coco_path, "w") as label_coco:
                    # Load the instance mask image; values are instance ids
                    img = np.array(
                        Image.open(os.path.join(mots_instances_dir, filename))
                    )

                    # Unique ids present in the mask (ignore 0 as background)
                    obj_ids = np.unique(img)[1:]
                    mask += len(obj_ids)
                    
                    # For each instance, compute bbox and write labels/gt
                    for obj_id in obj_ids:
                        # Binary mask for the current instance id
                        obj_mask = (img == obj_id).astype(np.uint8)

                        # If you wanted segmentation RLE, you'd call encode_mask_to_RLE(obj_mask)
                        # rle = encode_mask_to_RLE(obj_mask)

                        # Pixel indices (rows=y, cols=x) where mask=1
                        pos = np.where(obj_mask)
                        xmin = np.min(pos[1])
                        xmax = np.max(pos[1])
                        ymin = np.min(pos[0])
                        ymax = np.max(pos[0])
                        w = xmax - xmin
                        h = ymax - ymin

                        # Normalize instance ids to [1..1000] range for MOT
                        object_id = (obj_id % 1000) + 1
                        # Frame ids are 1-based here (filename assumed numeric)
                        frame_id = int(filename.split(".")[0]) + 1

                        # Image dimensions (fixed for AppleMOTS AppleMOT conversion)
                        mask_h = 972
                        mask_w = 1296

                        # YOLO normalized center x,y and width,height in [0,1]
                        nx = (xmin + xmax) / (2 * mask_w)
                        ny = (ymin + ymax) / (2 * mask_h)
                        nw = w / mask_w
                        nh = h / mask_h

                        # Examples of other output formats (kept as comments):
                        # print(frame_id)
                        # f.write(f'{filename},{obj_id},{xmin},{ymin},{xmax},{ymax}\n')
                        # f.write(f'{frame_id},{object_id},{xmin},{ymin},{w},{h}, 1, {rle}\n')

                        # For MOTS eval tool
                        # f.write(f'{frame_id},{object_id}, 1, {mask_h}, {mask_w}, {rle}\n')

                        # TrackEval-compatible MOT ground truth line:
                        # frame, id, x, y, w, h, 1, 1, 1
                        gt_mot.write(
                            f"{frame_id},{object_id},{xmin},{ymin},{w},{h},1,1,1\n"
                        )
                        
                        # Ultralytics/YOLO label: "<class> cx cy w h"
                        # Here, class id 47 is used for "apple".
                        label_coco.write("47 %.4f %.4f %.4f %.4f\n" % (nx, ny, nw, nh))
                    
                    # Track the maximum raw instance id seen
                    if np.max(obj_ids) > track:
                        track = np.max(obj_ids)
            else:
                # Non-image file: ignore and continue
                continue
    
    # Final bookkeeping: derive summary numbers
    track_id = track % 1000
    ls = [frames, track_id, mask]

    # Create seqinfo.ini next to the MOT 'img' and 'gt' folders.
    # NOTE: This uses string slicing 'mot_gt_dir[:-2]' to try to strip '/gt'.
    # If the path changes or does not end with 'gt', the slice will be incorrect.
    with open("%s/seqinfo.ini" % mot_gt_dir[:-2], "w") as seqinfo:
        seqinfo.write("[Sequence]\n")
        seqinfo.write("name=%s\n" % params["seq"])
        seqinfo.write("imDir=img\n")
        seqinfo.write("frameRate=30\n")
        seqinfo.write("seqLength=%d\n" % frames)
        seqinfo.write("imWidth=1296\n")
        seqinfo.write("imHeight=972\n")
        seqinfo.write("imExt=.%s\n" % ext)
    print("Done sequence %s" % params["seq"])
    return ls
