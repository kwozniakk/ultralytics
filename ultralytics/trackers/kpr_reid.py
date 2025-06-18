# Ultralytics 🚀 AGPL-3.0 License
"""KPR model wrapper for BoT-SORT re-identification."""

from typing import List

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
                "  pip install \"torchreid@git+https://github.com/victorjoos/keypoint_promptable_reidentification\""
            ) from e

        cfg = build_config(config_path=config_path)
        cfg.use_gpu = torch.cuda.is_available()
        self.extractor = KPRFeatureExtractor(cfg, verbose=False)

    def __call__(self, img: np.ndarray, dets: np.ndarray) -> List[np.ndarray]:
        """Return embeddings for given detections."""
        crops = [save_one_box(det, img, save=False) for det in xywh2xyxy(torch.from_numpy(dets[:, :4]))]
        samples = [{"image": crop} for crop in crops]
        _, embeddings, _, _ = self.extractor(samples)
        return [e.cpu().numpy() for e in embeddings]
