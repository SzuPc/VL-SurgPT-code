import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TissueVideoDataset(Dataset):
    def __init__(self, root_dir, folder_index=-1, max_frames=None):
        self.max_frames = max_frames
        self.sequences = []

        # 支持传入 case 目录或根目录
        if os.path.isdir(os.path.join(root_dir, "left")):
            case = Path(root_dir).name
            all_cases = [case]
            self.root_dir = str(Path(root_dir).parent)
        else:
            all_cases = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            if folder_index != -1:
                if folder_index >= len(all_cases):
                    raise ValueError(f"folder_index {folder_index} exceeds number of cases: {len(all_cases)}")
                all_cases = [all_cases[folder_index]]
            self.root_dir = root_dir

        for case in all_cases:
            case_path = os.path.join(self.root_dir, case)
            for side in ['left']:
                side_path = os.path.join(case_path, side)
                if not os.path.isdir(side_path):
                    continue
                for seq in os.listdir(side_path):
                    seq_path = os.path.join(side_path, seq)
                    video_path = os.path.join(seq_path, 'frames')
                    anno_path = os.path.join(seq_path, 'segmentation')
                    if not (os.path.exists(video_path) and os.path.exists(anno_path)):
                        continue

                    self.sequences.append({
                        'case': case,
                        'side': side,
                        'seq': seq,
                        'video_path': video_path,
                        'anno_path': anno_path
                    })

        print(f"Found {len(self.sequences)} valid sequences under {root_dir}")

    def _load_annotations(self, anno_path, scale_w, scale_h):
        # === 加载标签点 labels.json ===
        label_path = os.path.join(anno_path, "labels.json")
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Missing labels.json at {anno_path}")
        
        with open(label_path, 'r') as f:
            annotations = json.load(f)

        # === 加载文本信息 texts.json ===
        text_path = os.path.join(anno_path, "texts.json")
        if not os.path.exists(text_path):
            raise FileNotFoundError(f"Missing texts.json at {anno_path}")
        
        with open(text_path, 'r') as f:
            status_json = json.load(f)

        # === 初始化容器 ===
        frame_indices = sorted([int(k) for k in annotations.keys()])
        num_frames = max(frame_indices) + 1
        num_points = len(annotations.get("0", []))

        track_points = np.full((num_points, num_frames, 2), -1, dtype=np.float32)
        visibility = np.zeros((num_points, num_frames), dtype=bool)
        status_texts = [["" for _ in range(num_frames)] for _ in range(num_points)]
        location_texts = [["" for _ in range(num_frames)] for _ in range(num_points)]

        # === 遍历帧数据 ===
        for t in annotations:
            frame_idx = int(t)
            points = annotations[t]
            status_list = status_json.get(t, [])

            if points is None:
                continue

            for i, pt in enumerate(points):
                if pt is None or i >= len(status_list):
                    continue

                scaled_point = [pt[0] * scale_w, pt[1] * scale_h]
                track_points[i, frame_idx] = scaled_point
                visibility[i, frame_idx] = True

                # 提取状态与位置
                status_strs = status_list[i].get("status", [])
                location_str = status_list[i].get("location", "")
                if status_strs:
                    status_texts[i][frame_idx] = status_strs[0]
                if location_str:
                    location_texts[i][frame_idx] = location_str

        return track_points, visibility, status_texts, location_texts


    def _load_video_frames(self, video_path):
        files = os.listdir(video_path)
        video_files = [f for f in files if f.lower().endswith('.mp4')]

        if video_files:
            video_full_path = os.path.join(video_path, video_files[0])
        else:
            video_full_path = video_path  # 图像序列目录

        frames = []
        original_sizes = []

        for frame in get_video_frames(video_full_path):
            if frame is not None:
                original_height, original_width = frame.shape[:2]
                frame_resized = cv2.resize(frame, (720, 480))  # 标准尺寸
                frames.append(frame_resized)
                original_sizes.append((original_width, original_height))

        if not frames:
            raise RuntimeError(f"No frames loaded from {video_full_path}")

        original_width, original_height = original_sizes[0]
        scale_w = 720 / original_width
        scale_h = 480 / original_height

        return np.stack(frames), (scale_w, scale_h), original_sizes

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        try:
            seq_info = self.sequences[idx]
            frames, (scale_w, scale_h), original_sizes = self._load_video_frames(seq_info['video_path'])
            points, visibility, status_texts, location_texts = self._load_annotations(seq_info['anno_path'], scale_w, scale_h)

            # === 填充文本 ===
            status_texts = fill_missing_labels(status_texts, default="Unknown", stride=30)
            location_texts = fill_missing_labels(location_texts, default="Unknown", stride=30)

            frames = torch.from_numpy(frames).permute(0, 3, 1, 2).float()
            points = torch.from_numpy(points).float()
            visibility = torch.from_numpy(visibility)
            
            if self.max_frames is not None:
                frames = frames[:self.max_frames]
                points = points[:, :self.max_frames]
                visibility = visibility[:, :self.max_frames]
                status_texts = [s[:self.max_frames] for s in status_texts]
                location_texts = [l[:self.max_frames] for l in location_texts]

            return {
                'video': frames,                      # [T, C, H, W]
                'points': points,                     # [N, T, 2]
                'visibility': visibility,             # [N, T]
                'status_texts': status_texts,         # List[List[str]], fully filled
                "location_texts": location_texts,     # List[List[str]]
                'scale': (scale_w, scale_h),
                'original_sizes': original_sizes,
                'seq_info': {
                    'case': seq_info['case'],
                    'side': seq_info['side'],
                    'seq': seq_info['seq']
                }
            }
        except Exception as e:
            print(f"[Warning] Skipping idx {idx} due to error: {e}")
            return None



class GeneralVideoCapture(object):
    def __init__(self, path, reverse=False):
        images = Path(path).is_dir()
        self.image_inputs = images
        if images:
            self.path = path
            self.images = sorted([f for f in next(os.walk(path))[2]
                                  if os.path.splitext(f)[1].lower() in ['.jpg', '.png', '.jpeg']])
            if reverse:
                self.images = self.images[::-1]
            self.i = 0
        else:
            self.cap = cv2.VideoCapture(str(path))

    def read(self):
        if self.image_inputs:
            if self.i >= len(self.images):
                return False, None
            img_path = os.path.join(self.path, self.images[self.i])
            self.frame_src = self.images[self.i]
            img = cv2.imread(img_path)
            self.i += 1
            return True, img
        else:
            success, frame = self.cap.read()
            return success, frame

    def release(self):
        if not self.image_inputs:
            return self.cap.release()


def get_video_frames(path):
    cap = GeneralVideoCapture(path)
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break
        yield frame


def get_dataloader(root_dir, batch_size=1, num_workers=4, folder_index=-1, max_frames=None, collate_fn=None):
    dataset = TissueVideoDataset(root_dir, folder_index=folder_index, max_frames=max_frames)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return loader

def fill_missing_labels(text_grid, default="Unknown", stride=30):
    """
    修复版本：确保输出为 List[List[str]]，不再出现 List[List[List[str]]] 结构。
    - 支持原始值为 str 或 [str] 两种情况。
    - 保证每个位置都有填充结果。
    """
    N = len(text_grid)
    T = len(text_grid[0]) if N > 0 else 0
    filled = [["" for _ in range(T)] for _ in range(N)]

    for i in range(N):
        for start in range(0, T, stride):
            group_status = default
            # 找到当前 stride 内第一个有效 status
            for t in range(start, min(start + stride, T)):
                value = text_grid[i][t]
                if isinstance(value, list):
                    if len(value) > 0 and isinstance(value[0], str) and value[0].strip():
                        group_status = value[0].strip()
                        break
                elif isinstance(value, str) and value.strip():
                    group_status = value.strip()
                    break
            # 统一填充该段
            for t in range(start, min(start + stride, T)):
                filled[i][t] = group_status

    return filled
