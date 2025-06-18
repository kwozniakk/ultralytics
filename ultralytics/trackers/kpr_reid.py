# Ultralytics 🚀 AGPL-3.0 License
"""KPR model wrapper for BoT-SORT re-identification."""

from typing import List, Optional

import numpy as np
import torch

from ultralytics.utils.ops import xywh2xyxy
from ultralytics.utils.plotting import save_one_box


class KPRReID:
    """Wrapper around keypoint promptable re-identification (KPR) model."""

    def __init__(self, config_path: str):
        """Load KPR model from a configuration file."""
        try:
            from torchreid.scripts.builder import build_config
            from torchreid.tools.feature_extractor import KPRFeatureExtractor
        except ModuleNotFoundError as e:  # pragma: no cover - optional dependency
            raise ModuleNotFoundError(
                "KPR dependencies not found. Install them with\n"
                "  pip install \"torchreid@git+https://github.com/VlSomers/keypoint_promptable_reidentification\""
            ) from e

        cfg = build_config(config_path=config_path)
        cfg.use_gpu = torch.cuda.is_available()
        self.extractor = KPRFeatureExtractor(cfg, verbose=False)

    def __call__(
        self,
        img: np.ndarray,
        dets: np.ndarray,
        keypoints: Optional[np.ndarray] = None,
        negative_keypoints: Optional[np.ndarray] = None,
    ) -> List[np.ndarray]:
        """Return embeddings for given detections."""
        crops = [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        samples = []
        for i, crop in enumerate(crops):
            sample = {"image": crop}
            if keypoints is not None:
                sample["keypoints_xyc"] = np.asarray(keypoints[i])
            if negative_keypoints is not None:
                sample["negative_kps"] = np.asarray(negative_keypoints[i])
            samples.append(sample)

        _, embeddings, _, _ = self.extractor(samples)
        return [e.cpu().numpy() for e in embeddings]


def load_kpr_samples(images_folder: str, keypoints_folder: str) -> List[dict]:
    """Load sample dictionaries for the KPR model from image and keypoint folders."""
    import json
    import os
    import cv2

    image_files = [f for f in os.listdir(images_folder) if f.endswith(".jpg")]
    samples = []
    for img_name in image_files:
        img_path = os.path.join(images_folder, img_name)
        json_path = os.path.join(keypoints_folder, img_name.replace(".jpg", ".json"))

        img = cv2.imread(img_path)
        with open(json_path, "r") as json_file:
            keypoints_data = json.load(json_file)

        keypoints_xyc = []
        negative_kps = []
        for entry in keypoints_data:
            if entry["is_target"]:
                keypoints_xyc.append(entry["keypoints"])
            else:
                negative_kps.append(entry["keypoints"])

        assert len(keypoints_xyc) == 1, "Only one target keypoint set is supported for now."

        sample = {
            "image": img,
            "keypoints_xyc": np.array(keypoints_xyc[0]),
            "negative_kps": np.array(negative_kps),
        }
        samples.append(sample)

    return samples
