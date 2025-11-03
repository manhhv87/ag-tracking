import os
import sys
import shutil
import datetime

from ultralytics import YOLO
from ultralytics import settings


class UltraTrainer:
    """
    Thin convenience wrapper around Ultralytics YOLO (v8) for training, validation,
    and quick visual predictions. This class is used to produce/consume detector
    checkpoints that feed the tracking-by-detection pipeline described in the paper.

    Notes on how this maps to the paper's tracker:
    - The tracker relies on a robust detector to gate new-track creation (s_new)
      and keep-alive regression (s_active). Model capacity / checkpoint quality
      here will directly affect the ID stability discussed in the paper.
    - The `settings.update` call fixes dataset and runs directories so training,
      validation, and prediction stay reproducible and easy to locate.
    - `imgsz=640` establishes a consistent inference resolution; changing it
      changes the detector scale vs. the tracker’s geometric heuristics.
    """

    def __init__(self, cfg):
        """
        Initialize the trainer with a config dict and instantiate a YOLO model.

        Parameters
        ----------
        cfg : dict
            Expected keys:
              - "datadir": base dataset directory (Ultralytics will look for
                           `cfg['yolo_data']` paths relative to this when applicable).
              - "outdir": where Ultralytics saves runs (weights, logs, predictions).
              - "yolo_maker": path or name of a model spec/checkpoint, e.g. a YAML
                              (to build from scratch) or a .pt (to fine-tune/eval).
              - "yolo_data": dataset YAML (names/train/val/test).
              - "epochs": training epochs.
              - "batch_size": batch size.
              - "device": device string, e.g. "0" for a single GPU, "cpu", "0,1" etc.

        Behavior
        --------
        - Resets Ultralytics settings to defaults, then overrides datasets_dir + runs_dir.
        - Creates a YOLO object from the provided maker (YAML or checkpoint).
        """
        self.cfg = cfg
        # Reset any global Ultralytics runtime state to avoid path bleed-through
        settings.reset()
        # Pin the dataset home and the run outputs for reproducibility and clarity
        settings.update({"datasets_dir": cfg["datadir"], "runs_dir": cfg["outdir"]})
        # Build a model: if cfg["yolo_maker"] is a YAML, this initializes a new model
        # from architecture; if it's a .pt, it loads weights for finetune/eval.
        self.model = YOLO(
            cfg["yolo_maker"], verbose=True
        )  # build a new model from YAML file

    def train(self):
        """
        Launch YOLO training using the provided dataset YAML and hyper-params.

        Key arguments
        -------------
        data : cfg["yolo_data"]
            Points to a dataset YAML (Ultralytics format) that defines train/val
            image/label paths and class names.
        epochs : cfg["epochs"]
            Number of full passes through the training set.
        batch : cfg["batch_size"]
            Batches per step; adjust with VRAM.
        imgsz : 640
            Fixed input resolution; keep in sync with your downstream tracker’s
            expectations and any pre/post-processing (affects box scale stats).
        save_period : 1
            Save a checkpoint every epoch (useful for ablations/hyper sweeps).
        device : cfg["device"]
            "cpu", "0", "0,1", etc. depending on your environment.

        Impact on tracking
        ------------------
        Better detector recall/precision generally reduces ID switches and
        missed targets in the Tracktor/AGT association stage.
        """
        self.model.train(
            data=self.cfg["yolo_data"],
            epochs=self.cfg["epochs"],
            batch=self.cfg["batch_size"],
            imgsz=640,
            save_period=1,
            device=self.cfg["device"],
        )

    def eval(self):
        """
        Validate the current model (self.model) on the dataset specified by
        the model's training configuration or default Ultralytics behavior.

        Behavior
        --------
        - Prints a warning tag if using the 'yolov8n.pt' (nano) checkpoint to
          remind that it's a very small capacity model (speed > accuracy).
        - Runs `model.val()` which computes mAP and other standard metrics.

        Tip
        ---
        Align evaluation set with the distribution your tracker will see (same
        crops/lighting/occlusion patterns as in the paper’s scenarios).
        """
        warn = ""
        if self.cfg["yolo_maker"] == "yolov8n.pt":
            warn = "WARNING:"
        print(warn + " Using checkpoint:", self.cfg["yolo_maker"])
        self.model.val()

    def predict(self, max=50, img=None):
        """
        Run quick predictions for qualitative checks.

        Parameters
        ----------
        max : int
            Maximum number of images to run when `img` is not provided.
        img : Optional[str or path-like]
            If given, run prediction on that single image; otherwise iterate over
            `cfg['datadir']/testing/images/` and save outputs.

        Behavior
        --------
        - If a single `img` is passed, results are displayed in the console and
          saved under the Ultralytics runs directory.
        - Otherwise, loops through up to `max` images in the testing folder and
          saves prediction images with boxes (useful for sanity-checking labels
          and the checkpoint before plugging into the tracker).

        Side effect
        -----------
        Prints the active Ultralytics runs directory at the end (where artifacts
        like predictions and weights are stored).
        """
        if img:
            # Single-image prediction; Ultralytics handles loading/visualization/saving
            self.model(img)
        else:
            # Batch quicklook over a testing images folder for qualitative review
            for i, file in enumerate(
                os.listdir(self.cfg["datadir"] + "testing/images/")
            ):
                if i == max:
                    break
                if file.endswith(".jpg") or file.endswith(".png"):
                    self.model(
                        self.cfg["datadir"] + "testing/images/" + file,
                        save=True,
                        verbose=False,
                    )
        
        # Show where the artifacts (predictions) were written
        print("Results saved to", settings("runs_dir"))
