# Ultralytics 🚀 AGPL-3.0 License - https://ultralytics.com/license
"""Body part-based tracker using BoT-SORT with pose ReID."""

from typing import Any, List

import numpy as np
import torch

from ultralytics.utils.plotting import save_one_box
from ultralytics.utils.ops import xywh2xyxy

from .bot_sort import BOTSORT


class BodyPartReID:
    """Simplified body part-based ReID using a pose model."""

    PART_IDXS = [[0, 1, 2, 3, 4], [5, 6, 7, 8, 9, 10, 11], [12, 13, 14, 15, 16]]

    def __init__(self, model: str = "yolo11n-pose.pt"):
        from ultralytics import YOLO

        self.model = YOLO(model)
        # warmup to create predictor
        self.model.predict([np.zeros((10, 10, 3), dtype=np.uint8)], verbose=False)

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> List[np.ndarray]:
        crops = [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        results = self.model(crops, verbose=False)
        feats = []
        for r in results:
            if getattr(r, "keypoints", None) is not None and len(r.keypoints.xyn):
                kpts = r.keypoints.xyn[0].cpu().numpy()
                parts = []
                for idxs in self.PART_IDXS:
                    p = kpts[idxs, :2]
                    p = p[p.sum(1) != 0]  # filter missing
                    parts.append(p.mean(0) if len(p) else np.zeros(2))
                f = np.concatenate(parts).astype(np.float32)
            else:
                f = np.zeros(6, dtype=np.float32)
            n = np.linalg.norm(f)
            feats.append(f / n if n > 0 else f)
        return feats


class BPBreIDSORT(BOTSORT):
    """BoT-SORT tracker with body part-based ReID."""

    def __init__(self, args: Any, frame_rate: int = 30):
        super().__init__(args, frame_rate)
        if args.with_reid:
            self.encoder = BodyPartReID(args.model if args.model != "auto" else "yolo11n-pose.pt")
