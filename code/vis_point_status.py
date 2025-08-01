import os
import cv2
import json
from pathlib import Path

# === 配置路径 ===

video_dir = Path("/home/lenovo/acm_mm_dataset/instrument_text_guidance_dataset/test/0/left/seq003/frames")  # 包含 mp4 的文件夹
json_path = Path("/home/lenovo/acm_mm_dataset/vis_track_status/instrument/0/left/seq003/point_and_status.json")
output_image_dir = Path("/home/lenovo/acm_mm_dataset/vis_track_status/instrument/0/left/seq003/output_frames")
output_video_path = Path("/home/lenovo/acm_mm_dataset/vis_track_status/instrument/0/left/seq003/output_video.mp4")

output_image_dir.mkdir(parents=True, exist_ok=True)

# === 加载 JSON ===
with open(json_path, "r") as f:
    frame_data = json.load(f)
frame_index_to_data = {item["frame_index"]: item for item in frame_data}

# === 查找视频 ===
video_files = list(video_dir.glob("*.mp4"))
if len(video_files) != 1:
    raise RuntimeError(f"Expected one .mp4 file, found {len(video_files)}")
video_path = str(video_files[0])

# === 打开视频并逐帧处理 ===
cap = cv2.VideoCapture(video_path)
frame_idx = 1  # 如果你的 JSON 是从 frame 1 开始标注的

success, frame = cap.read()
frame_height, frame_width = frame.shape[:2]
cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # 回到第一帧

fourcc = cv2.VideoWriter_fourcc(*'mp4v')
video_writer = cv2.VideoWriter(str(output_video_path), fourcc, cap.get(cv2.CAP_PROP_FPS), (frame_width, frame_height))

while True:
    success, frame = cap.read()
    if not success:
        break

    if frame_idx in frame_index_to_data:
        data = frame_index_to_data[frame_idx]
        coords = data["coordinates"]
        statuses = data["status_preds"]

        for (x, y), status in zip(coords, statuses):
            x_int, y_int = int(round(x)), int(round(y))
            cv2.circle(frame, (x_int, y_int), 4, (0, 0, 255), -1)
            cv2.putText(frame, status, (x_int + 5, y_int), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # 保存帧为图像
    frame_filename = f"{frame_idx:04d}.png"
    cv2.imwrite(str(output_image_dir / frame_filename), frame)

    # 写入视频
    video_writer.write(frame)

    frame_idx += 1

cap.release()
video_writer.release()

print(f"==> 保存图片到: {output_image_dir}")
print(f"==> 合成视频到: {output_video_path}")
