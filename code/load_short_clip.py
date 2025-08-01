import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class GeneralVideoCapture:
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
            img = cv2.imread(img_path)
            self.i += 1
            return True, img
        else:
            return self.cap.read()

    def release(self):
        if not self.image_inputs:
            self.cap.release()


def get_video_frames(path):
    cap = GeneralVideoCapture(path)
    while True:
        success, frame = cap.read()
        if not success or frame is None:
            break
        yield frame


def fill_missing_labels(text_grid):
    """
    Args:
        text_grid: List[List[str]]，大小为 [N, T]，每个元素为状态字符串或空字符串。
    Returns:
        filled: List[List[str]]，仅在空字符串时向前填充，不会向后找。
    """
    N = len(text_grid)
    T = len(text_grid[0]) if N > 0 else 0
    filled = [["" for _ in range(T)] for _ in range(N)]

    for i in range(N):
        for t in range(T):
            if text_grid[i][t] and text_grid[i][t].strip().lower() != "unknown":
                # 当前帧有标签，直接使用
                filled[i][t] = text_grid[i][t].strip()
            elif t > 0:
                # 当前为空，则填充上一帧的值（无论上一帧是否为 unknown）
                filled[i][t] = filled[i][t - 1]

    return filled



class TissueVideoDataset(Dataset):
    def __init__(self, root_dir, folder_index=-1, clip_len=31):
        self.clip_len = clip_len
        self.clips = []
        self.seq_annotations = {}

        if os.path.isdir(os.path.join(root_dir, "left")):
            case = Path(root_dir).name
            all_cases = [case]
            self.root_dir = str(Path(root_dir).parent)
        else:
            all_cases = sorted([d for d in os.listdir(root_dir) if os.path.isdir(os.path.join(root_dir, d))])
            if folder_index != -1:
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

                    video_files = [f for f in os.listdir(video_path) if f.lower().endswith('.mp4')]
                    if not video_files:
                        continue
                    video_full_path = os.path.join(video_path, video_files[0])

                    frames = []
                    original_sizes = []
                    for frame in get_video_frames(video_full_path):
                        if frame is not None:
                            h, w = frame.shape[:2]
                            frame_resized = cv2.resize(frame, (720, 480))
                            frames.append(frame_resized)
                            original_sizes.append((w, h))
                    if len(frames) == 0:
                        continue

                    frames_np = np.stack(frames)
                    scale_w = 720 / original_sizes[0][0]
                    scale_h = 480 / original_sizes[0][1]

                    points, visibility, status_texts, location_texts = self._load_annotations(
                        anno_path, scale_w, scale_h, len(frames)
                    )
                    status_texts = fill_missing_labels(status_texts)
                    location_texts = fill_missing_labels(location_texts)

                    key = (case, side, seq)
                    self.seq_annotations[key] = {
                        'frames': frames_np,
                        'points': points,
                        'visibility': visibility,
                        'status_texts': status_texts,
                        'location_texts': location_texts,
                        'original_sizes': original_sizes,
                        'scale': (scale_w, scale_h)
                    }

                    for start in range(0, len(frames) - clip_len + 1, clip_len - 1):
                        self.clips.append({
                            'key': key,
                            'start': start,
                            'end': start + clip_len
                        })

    def _load_annotations(self, anno_path, scale_w, scale_h, total_frames):
        
        label_path = os.path.join(anno_path, "labels.json")
        with open(label_path, 'r') as f:
            annotations = json.load(f)

        text_path = os.path.join(anno_path, "texts.json")
        with open(text_path, 'r') as f:
            status_json = json.load(f)

        num_points = len(annotations.get("0", []))
        track_points = np.full((num_points, total_frames, 2), -1, dtype=np.float32)
        visibility = np.zeros((num_points, total_frames), dtype=bool)
        status_texts = [["" for _ in range(total_frames)] for _ in range(num_points)]
        location_texts = [["" for _ in range(total_frames)] for _ in range(num_points)]

        for t_str in annotations:
            t = int(t_str)
            if t >= total_frames:
                continue
            points = annotations[t_str]
            status_list = status_json.get(t_str, [])
            if points is None:
                continue
            for i, pt in enumerate(points):
                if pt is None or i >= len(status_list):
                    continue
                track_points[i, t] = [pt[0] * scale_w, pt[1] * scale_h]
                visibility[i, t] = True
                entry = status_list[i] if isinstance(status_list[i], dict) else {}
                status = entry.get("status", [""])
                location = entry.get("location", [""])
                status_texts[i][t] = status[0] if isinstance(status, list) and status else ""
                location_texts[i][t] = location[0] if isinstance(location, list) and location else ""

        return track_points, visibility, status_texts, location_texts

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        try:
            clip = self.clips[idx]
            key = clip['key']
            info = self.seq_annotations[key]
            s, e = clip['start'], clip['end']
            frames = torch.from_numpy(info['frames'][s:e]).permute(0, 3, 1, 2).float() / 255.0
            points = torch.from_numpy(info['points'][:, s:e]).float()
            visibility = torch.from_numpy(info['visibility'][:, s:e])

            return {
                'video': frames,
                'points': points,
                'visibility': visibility,
                'status_texts': [row[s:e] for row in info['status_texts']],
                'location_texts': [row[s:e] for row in info['location_texts']],
                'scale': info['scale'],
                'original_sizes': info['original_sizes'][s:e],
                'seq_info': {
                    'case': key[0],
                    'side': key[1],
                    'seq': key[2]
                }
            }
        except Exception as e:
            print(f"[Warning] Failed to load idx {idx}: {e}")
            return None


def get_dataloader(root_dir, batch_size=1, num_workers=4, folder_index=-1, clip_len=31, collate_fn=None):
    dataset = TissueVideoDataset(root_dir, folder_index=folder_index, clip_len=clip_len)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True, collate_fn=collate_fn)
    return loader
