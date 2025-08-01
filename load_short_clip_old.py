import os
import json
import torch
import numpy as np
import cv2
from torch.utils.data import Dataset, DataLoader
from pathlib import Path


class TissueVideoDataset(Dataset):
    def __init__(self, root_dir, folder_index=-1, clip_len=31):
        self.clip_len = clip_len
        self.clips = []

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

                    video_files = [f for f in os.listdir(video_path) if f.lower().endswith('.mp4')]
                    if not video_files:
                        continue

                    video_full_path = os.path.join(video_path, video_files[0])
                    cap = cv2.VideoCapture(video_full_path)
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    cap.release()

                    for start in range(0, frame_count - self.clip_len + 1, self.clip_len - 1):
                        self.clips.append({
                            'case': case,
                            'side': side,
                            'seq': seq,
                            'video_path': video_path,
                            'anno_path': anno_path,
                            'start': start,
                            'end': start + self.clip_len
                        })

        print(f"Found {len(self.clips)} clips under {root_dir}")
        
        


    def _load_annotations(self, anno_path, scale_w, scale_h, start, end):
        label_path = os.path.join(anno_path, "labels.json")
        with open(label_path, 'r') as f:
            annotations = json.load(f)

        text_path = os.path.join(anno_path, "texts.json")
        with open(text_path, 'r') as f:
            status_json = json.load(f)

        num_points = len(annotations.get("0", []))
        num_frames = end - start

        track_points = np.full((num_points, num_frames, 2), -1, dtype=np.float32)
        visibility = np.zeros((num_points, num_frames), dtype=bool)
        status_texts = [["" for _ in range(num_frames)] for _ in range(num_points)]
        location_texts = [["" for _ in range(num_frames)] for _ in range(num_points)]

        for t_str in annotations:
            t = int(t_str)
            if t < start or t >= end:
                continue
            frame_idx = t - start
            points = annotations[t_str]
            status_list = status_json.get(t_str, [])

            if points is None:
                continue

            for i, pt in enumerate(points):
                if pt is None or i >= len(status_list):
                    continue
                scaled_point = [pt[0] * scale_w, pt[1] * scale_h]
                track_points[i, frame_idx] = scaled_point
                visibility[i, frame_idx] = True

                entry = status_list[i] if i < len(status_list) and isinstance(status_list[i], dict) else {}
                status_list_raw = entry.get("status", [])
                location_list_raw = entry.get("location", [])

                # Ensure it's a non-empty string
                status_str = status_list_raw[0] if isinstance(status_list_raw, list) and status_list_raw else ""
                location_str = location_list_raw[0] if isinstance(location_list_raw, list) and location_list_raw else ""

                status_texts[i][frame_idx] = status_str
                location_texts[i][frame_idx] = location_str


        return track_points, visibility, status_texts, location_texts

    def _load_video_frames(self, video_path, start=None, end=None):
        video_files = [f for f in os.listdir(video_path) if f.lower().endswith('.mp4')]
        video_full_path = os.path.join(video_path, video_files[0])

        frames = []
        original_sizes = []

        for i, frame in enumerate(get_video_frames(video_full_path)):
            if i < start:
                continue
            if end is not None and i >= end:
                break
            if frame is not None:
                h, w = frame.shape[:2]
                frame_resized = cv2.resize(frame, (720, 480))
                frames.append(frame_resized)
                original_sizes.append((w, h))

        if len(original_sizes) == 0:
            raise ValueError(f"No frames found in range {start}-{end} from {video_full_path}")

        original_width, original_height = original_sizes[0]
        scale_w = 720 / original_width
        scale_h = 480 / original_height

        return np.stack(frames), (scale_w, scale_h), original_sizes

    def __len__(self):
        return len(self.clips)

    def __getitem__(self, idx):
        try:
            clip = self.clips[idx]
            start, end = clip['start'], clip['end']
            frames_all, (scale_w, scale_h), original_sizes = self._load_video_frames(
                clip['video_path'], start=start, end=end
            )
            frames = torch.from_numpy(frames_all).permute(0, 3, 1, 2).float()

            points, visibility, status_texts, location_texts = self._load_annotations(
                clip['anno_path'], scale_w, scale_h, start, end
            )

            status_texts = fill_missing_labels(status_texts)
            location_texts = fill_missing_labels(location_texts)


            points = torch.from_numpy(points).float()
            visibility = torch.from_numpy(visibility)

            return {
                'video': frames,
                'points': points,
                'visibility': visibility,
                'status_texts': status_texts,
                'location_texts': location_texts,
                'scale': (scale_w, scale_h),
                'original_sizes': original_sizes,
                'seq_info': {
                    'case': clip['case'],
                    'side': clip['side'],
                    'seq': clip['seq']
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


def get_dataloader(root_dir, batch_size=1, num_workers=4, folder_index=-1, clip_len=31, collate_fn=None):
    dataset = TissueVideoDataset(root_dir, folder_index=folder_index, clip_len=clip_len)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    return loader


def fill_missing_labels(text_grid, stride=30):
    N = len(text_grid)
    T = len(text_grid[0]) if N > 0 else 0
    filled = [["" for _ in range(T)] for _ in range(N)]

    for i in range(N):
        last_valid = ""
        for t in range(T):
            if text_grid[i][t]:
                last_valid = text_grid[i][t]
            filled[i][t] = last_valid

        # 如果全部为空（last_valid仍为""），可以往后找一段再填回来
        if not any(filled[i]):
            for t in range(T - 1, -1, -1):
                if text_grid[i][t]:
                    for k in range(T):
                        filled[i][k] = text_grid[i][t]
                    break
    return filled


