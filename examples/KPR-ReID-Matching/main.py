#!/usr/bin/env python
"""Simple script to compare person embeddings using KPR."""

import argparse
import os
from typing import List

import numpy as np
from ultralytics.trackers.kpr_reid import KPRReID, load_kpr_samples


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="KPR ReID demo")
    parser.add_argument("--config", required=True, help="Path to KPR config YAML")
    parser.add_argument(
        "--data",
        default="ultralytics/assets/demo/soccer_players",
        help=(
            "Folder with images/ and keypoints/ subfolders. Only keypoint JSONs"
            " are included; add your own images."
        ),
    )
    parser.add_argument(
        "--reference", default="personA1.jpg", help="Reference image name inside the images folder"
    )
    return parser.parse_args()


def get_sample_by_name(samples: List[tuple], name: str) -> dict:
    for s, fname in samples:
        if fname == name:
            return s
    raise FileNotFoundError(name)


def main(args: argparse.Namespace) -> None:
    encoder = KPRReID(args.config)

    images_folder = os.path.join(args.data, "images")
    kps_folder = os.path.join(args.data, "keypoints")

    # Load samples and keep file names
    image_files = sorted(f for f in os.listdir(images_folder) if f.endswith(".jpg"))
    samples_only = load_kpr_samples(images_folder, kps_folder)
    samples = list(zip(samples_only, image_files))

    ref_sample = get_sample_by_name(samples, args.reference)

    dets = np.array([[48, 80, 96, 160]])  # center x,y,w,h covering full image
    ref_emb = encoder(ref_sample["image"], dets, [ref_sample["keypoints_xyc"]], [ref_sample["negative_kps"]])[0]

    print(f"Reference embedding from {args.reference}")
    for s, fname in samples:
        emb = encoder(s["image"], dets, [s["keypoints_xyc"]], [s["negative_kps"]])[0]
        dist = np.linalg.norm(ref_emb - emb)
        print(f"{fname}: distance={dist:.3f}")


if __name__ == "__main__":
    main(parse_args())
