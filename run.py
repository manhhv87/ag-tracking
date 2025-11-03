"""
run.py
======

Entry-point script for running different tasks across Agricultural MOT-style datasets.

Supported tasks (via config `cfg["task"]`)
------------------------------------------
- "play":   Render / visualize sequences interactively using `render_utils.play`.
- "det":    Detection workflow; trains/evaluates either Faster R-CNN (FRCNN_FPN)
            or YOLOv8 (Ultralytics) depending on `cfg["detector"]`.
- "track":  Tracking workflow; supports `byte_track`, generic `track`, or SLAM-based tracking.
- "slam":   SLAM-specific pipeline for sequence processing.

Datasets & configs
------------------
- Datasets are wrapped in `AgMOT` (per-set) and can be merged via `MergedAgMOT`
  for training/evaluation. Two pre-defined configs are provided:
  * `cfg_lettuce` for LettuceMOT (from `config.lettuce_config`)
  * `cfg_apples`  for AppleMOT/AppleMOTS (from `config.apples_config`)
- The dataset selection is controlled by CLI arg `--dataset` (default "AppleMOTS").

Safety gate
-----------
- Before performing any heavyweight action (loading data, writing results), the
  script prompts the user with `continue? (y/N):` to avoid accidental runs
  that might modify data or produce large outputs.

High-level flow
---------------
1) Parse CLI arg `--dataset` and choose a base config.
2) Set run-time controls: `batch`, `shuffle`, and `train` flags from config.
3) Build one `AgMOT` dataset + `DataLoader` per set listed in `cfg["sets"]`.
4) Dispatch to the selected `task`:
   - play:  call `play(cfg, dataloader, name)` for each set
   - det:   instantiate trainer (FRCNN_FPN or YOLOv8) and train/eval/predict
   - slam:  call `slam(cfg, dataloader, name)` and ensure output dirs exist
   - track: run `track_eval` or `track/byte_track` depending on mode & dataset
"""

import os
import tqdm
import argparse
from ultralytics import YOLO
from lib.utils.render_utils import play
from torch.utils.data import DataLoader
from lib.utils.track_eval import track_eval
from lib.models.tracker import byte_track, track, slam
from lib.trainers.frcnn_trainer import FrcnnTrainer
from lib.trainers.yolo_trainer import UltraTrainer
from lib.datasets.ag_mot import AgMOT, MergedAgMOT

from config.lettuce_config import cfg as cfg_lettuce
from config.apples_config import cfg as cfg_apples

if __name__ == "__main__":
    # -----------------------------
    # 1) Parse dataset CLI argument
    # -----------------------------
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-d", "--dataset", type=str, default="AppleMOTS", help="Dataset to use"
    )
    args = parser.parse_args()

    # Default benchmark label; will be changed for Lettuce.
    benchmark = "AppleMOT"

    # Choose configuration based on dataset name.
    # Note: case-insensitive substring checks for convenience.
    if "lettuce" in args.dataset.lower():
        cfg = cfg_lettuce
        benchmark = "LettuceMOT"
    if "apple" in args.dataset.lower():
        cfg = cfg_apples
        # For Apple, adjust paths to MOT directory layout.
        cfg["datadir"] += "MOT/"
        cfg["outdir"] += "MOT/"

    print("Collection:", benchmark)

    # ---------------------------------------------------
    # 2) Derive batch size / shuffling / training switch
    # ---------------------------------------------------
    # For interactive or sequence-processing tasks, force batch=1 and no shuffle.
    batch = 1 if cfg["task"] in ["play", "track", "slam"] else cfg["batch_size"]
    shuffle = False if cfg["task"] in ["play", "track", "slam"] else True

    # Training flag is only true for detection in train mode.
    train = cfg["task"] == "det" and cfg["mode"] == "train"

    # --------------------------------------------
    # 3) Report task/mode/sets and safety prompt
    # --------------------------------------------
    print(
        "| task:", cfg["task"], "| mode:", cfg["mode"], "| datasets:", cfg["sets"], "|"
    )

    # SAFETY GATE: ask for user confirmation before proceeding.
    # This avoids expensive or destructive operations due to accidental runs.
    inp = input("continue? (y/N): ")
    if inp.lower() != "y":
        exit()

    # --------------------------------------------
    # 4) Build datasets and dataloaders per set
    # --------------------------------------------
    datasets, dataloaders = [], []
    for i, seti in enumerate(cfg["sets"]):
        # Each AgMOT instance wraps MOT-style sequences for a particular split/set.
        datasets.append(AgMOT(cfg, seti, train=train))

        # Construct a DataLoader with the derived batch/shuffle and configured workers.
        dataloaders.append(
            DataLoader(
                datasets[i],
                batch_size=batch,
                shuffle=shuffle,
                num_workers=cfg["num_workers"],
            )
        )

    # --------------------------------------------
    # 5) Dispatch to the selected pipeline (task)
    # --------------------------------------------

    # ---- Visualization / playback mode ----
    if cfg["task"] == "play":
        for dataloader, name in zip(dataloaders, cfg["sets"]):
            print("\n" * 2 + "dataset:", name)
            play(cfg, dataloader, name)

    # ---- Detection (train/eval/predict) ----
    if cfg["task"] == "det":
        # Special-case: YOLOv8 + Apple -> use COCO directory layout instead of MOT.
        if cfg["detector"] == "yolov8" and "apple" in args.dataset.lower():
            cfg["datadir"] = cfg["datadir"].replace("/MOT/", "/COCO/")
            cfg["outdir"] = cfg["outdir"].replace("/MOT/", "/COCO/")

        # Optionally merge multiple AgMOT datasets into a single training set.
        merged = MergedAgMOT(datasets)

        # Choose the appropriate trainer based on the detector type.
        if cfg["detector"] == "frcnn":
            trainer = FrcnnTrainer(cfg, merged)
        elif cfg["detector"] == "yolov8":
            trainer = UltraTrainer(cfg)
        else:
            raise ValueError("Invalid detector")

        # Execute the requested mode for detection.
        if cfg["mode"] == "train":
            trainer.train()
        if cfg["mode"] == "eval":
            trainer.eval()
        if cfg["mode"] == "predict":
            trainer.predict()

    # ---- SLAM processing ----
    if cfg["task"] == "slam":
        for dataloader, name in zip(dataloaders, cfg["sets"]):
            print("\n" * 2 + "dataset:", name)
            # Ensure the output directory exists for SLAM artifacts.
            os.makedirs(cfg["outdir"] + "slam", exist_ok=True)
            slam(cfg, dataloader, name)

    # ---- Tracking (evaluation or prediction) ----
    if cfg["task"] == "track":
        if cfg["mode"] == "eval":
            # Evaluate tracker outputs using the benchmark protocol (MOT metrics).
            track_eval(cfg, benchmark)
        elif cfg["mode"] == "predict":
            # Guardrail: ByteTrack currently unsupported for LettuceMOT.
            if cfg["tracker"] == "bytetrack" and "lettuce" in args.dataset.lower():
                raise ValueError(
                    "ByteTrack tracker is not yet available for LettuceMOT"
                )

            # For Apple datasets and ByteTrack, run the dedicated multi-seq driver.
            if cfg["tracker"] == "bytetrack" and "apple" in args.dataset.lower():
                byte_track(cfg, datasets)
            else:
                # Generic tracking: process each set independently and persist results.
                for dataloader, name in zip(dataloaders, cfg["sets"]):
                    print("\n" * 2 + "dataset:", name)
                    os.makedirs(
                        cfg["outdir"] + "trackers/" + cfg["tracker"], exist_ok=True
                    )
                    track(cfg, dataloader, name)
        else:
            # Explicit error to guide users to supported modes under "track".
            raise ValueError(
                "Invalid mode: "
                + cfg["mode"]
                + "for task: track. Use eval or predict. Training is not yet available."
            )
