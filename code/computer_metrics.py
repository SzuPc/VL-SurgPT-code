import numpy as np
from pathlib import Path
import json
import torch
import logging

logger = logging.getLogger(__name__)

class TrackingEvaluator:
    def __init__(self, abnormal_error_threshold=500):
        self.sequence_results = []
        self.abnormal_error_threshold = abnormal_error_threshold

    def _compute_distance_metrics(self, pred_pts, gt_pts):
        point_errors = []
        for i in range(len(pred_pts)):
            pt = gt_pts[i]

            # 过滤非法 ground truth（例如 [-3.75, -2.8125] 或任意小于 0 的坐标）
            if pt is None or not isinstance(pt, (list, tuple, np.ndarray)) or np.any(np.array(pt) < 0):
                point_errors.append(None)
                continue

            error = np.linalg.norm(pred_pts[i] - pt)
            if error > self.abnormal_error_threshold:
                print(f"[Warning] Abnormal error at point {i}: pred={pred_pts[i]}, gt={pt}, error={error:.2f}")
                point_errors.append(None)
            else:
                point_errors.append(float(error))
        return point_errors

    def compute_sequence_error(self, pred_points, gt_points, visibility, seq_info):
        num_points = gt_points.shape[0]
        num_frames = gt_points.shape[1]
        points_all_errors = {i: {} for i in range(num_points)}
        points_mean_errors = {i: [] for i in range(num_points)}

        for t in range(num_frames):
            curr_pred = pred_points[t][0].cpu().numpy() if isinstance(pred_points[t][0], torch.Tensor) else pred_points[t][0]
            curr_gt = gt_points[:, t].cpu().numpy() if isinstance(gt_points, torch.Tensor) else gt_points[:, t]

            errors = self._compute_distance_metrics(curr_pred, curr_gt)

            for i in range(num_points):
                if errors[i] is not None:
                    points_all_errors[i][str(t)] = errors[i]
                    if visibility[i, t] and errors[i] > 0:
                        points_mean_errors[i].append(errors[i])
                else:
                    points_all_errors[i][str(t)] = None

        point_average_errors = {
            i: float(np.mean(errors)) if errors else 0.0
            for i, errors in points_mean_errors.items()
        }

        return {
            'frame_errors': points_all_errors,
            'mean_errors': point_average_errors
        }

def save_sequence_results(results, errors, save_dir, model_name):
    """保存序列结果"""
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    # 保存 tracking 结果
    tracking_results = {}
    for t, (coords, occlusions) in enumerate(results):
        if isinstance(coords, torch.Tensor):
            coords = coords.cpu().numpy()
        if isinstance(occlusions, torch.Tensor):
            occlusions = occlusions.cpu().numpy()

        tracking_results[str(t)] = {
            "coordinates": coords.tolist(),
            "occlusions": occlusions.tolist()
        }

    # 找到最后一帧编号
    all_frame_ids = []
    for frame_dict in errors['frame_errors'].values():
        all_frame_ids.extend(map(int, frame_dict.keys()))
    max_frame_id = max(all_frame_ids)

    # 只保存每 30 帧和最后一帧
    filtered_frame_errors = {}
    for pt_idx, frame_dict in errors['frame_errors'].items():
        selected_frames = {}
        for fid_str, err in frame_dict.items():
            fid = int(fid_str)
            if fid % 30 == 0 or fid == max_frame_id:
                selected_frames[str(fid)] = err if err is not None else float('nan')
        if selected_frames:
            filtered_frame_errors[str(pt_idx)] = selected_frames

    output = {
        "tracking_results": tracking_results,
        "metrics": {
            "frame_errors": filtered_frame_errors,
            "point_mean_errors": errors['mean_errors'],
            "overall_mean_error": float(np.mean([v for v in errors['mean_errors'].values() if v > 0]))
        }
    }

    with open(save_dir / f"{model_name}_result.json", "w") as f:
        json.dump(output, f, indent=2, allow_nan=True)
