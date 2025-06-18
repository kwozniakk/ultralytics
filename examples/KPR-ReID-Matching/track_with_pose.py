#!/usr/bin/env python
"""Track people with YOLO pose and KPR re-identification."""

import argparse

import numpy as np
from ultralytics import YOLO
from ultralytics.utils import IterableSimpleNamespace
from ultralytics.trackers.bot_sort import BOTSORT


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KPR tracking demo")
    parser.add_argument("--kpr", required=True, help="Path to KPR config YAML")
    parser.add_argument("--source", default="ultralytics/assets/bus.jpg", help="Image or video path")
    parser.add_argument("--pose-model", default="yolov8n-pose.pt", help="YOLO pose model")
    return parser.parse_args()


def main(args: argparse.Namespace) -> None:
    model = YOLO(args.pose_model)
    cfg = IterableSimpleNamespace(
        tracker_type="botsort",
        with_reid=True,
        model=args.kpr,
        track_high_thresh=0.25,
        track_low_thresh=0.1,
        new_track_thresh=0.25,
        track_buffer=30,
        match_thresh=0.8,
        fuse_score=True,
        gmc_method="sparseOptFlow",
        proximity_thresh=0.5,
        appearance_thresh=0.8,
    )
    tracker = BOTSORT(cfg)

    for result in model.predict(args.source, stream=True):
        boxes = result.boxes.cpu().numpy()
        if result.keypoints is not None:
            kpts = result.keypoints.cpu().numpy()
            neg = [np.delete(kpts, i, axis=0) for i in range(len(kpts))]
            extra = {"keypoints": kpts, "negative_keypoints": neg}
        else:
            extra = None
        tracks = tracker.update(boxes, result.orig_img, extra)
        print("Frame", result.path, "tracks", tracks)


if __name__ == "__main__":
    main(parse_args())

