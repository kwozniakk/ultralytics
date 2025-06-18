# KPR Re-Identification Matching Demo

This example shows how to use the optional [Keypoint Promptable Re-Identification](https://github.com/VlSomers/keypoint_promptable_reidentification) (KPR) module with Ultralytics trackers.
It compares embeddings from a few example keypoint annotations using a KPR model. Only keypoint JSON files are
included in this repository; you need to supply matching images yourself.
Each sample dictionary includes the positive keypoints for the target person (`keypoints_xyc`) and
keypoints from other people in the image (`negative_kps`).

## Usage

1. Install KPR dependencies (see documentation):
   ```bash
   pip install "torchreid@git+https://github.com/VlSomers/keypoint_promptable_reidentification"
   ```

2. Run the script with a KPR configuration file:
   ```bash
   python main.py --config path/to/kpr_config.yaml
   ```
   Optional arguments:
   - `--data`: path to a folder containing `images/` and `keypoints/` subfolders. Sample keypoints live under
     `ultralytics/assets/demo/soccer_players/keypoints`, but you must provide the corresponding images.
   - `--reference`: image file name to use as the reference person.

3. Track with YOLO pose and KPR:
   ```bash
   python track_with_pose.py --kpr path/to/kpr_config.yaml --source path/to/video.mp4
   ```
   This script forms negative prompts from other detections in each frame.
