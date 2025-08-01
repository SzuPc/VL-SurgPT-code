import os
import json

def find_all_status_fields(json_root):
    print(f"开始遍历目录: {json_root}\n")
    unknown_count = 0
    total_status_count = 0

    for root, _, files in os.walk(json_root):
        for file in files:
            if file.endswith(".json"):
                file_path = os.path.join(root, file)
                try:
                    with open(file_path, "r") as f:
                        data = json.load(f)

                    if not isinstance(data, dict):
                        print(f"[跳过] 非 dict 类型 JSON 文件: {file_path}")
                        continue

                    for frame_idx, entries in data.items():
                        if not isinstance(entries, list):
                            continue

                        for entry in entries:
                            if isinstance(entry, dict) and "status" in entry:
                                status = entry["status"]
                                total_status_count += 1

                                if status == "Unknown":
                                    unknown_count += 1
                                    print(f"[发现 Unknown] 文件: {file_path}\n  帧号: {frame_idx}\n  状态项: {entry}\n")
                                else:
                                    print(f"[status] {status} - 来自: {file_path}, 帧: {frame_idx}")

                except Exception as e:
                    print(f"[错误] 无法读取文件: {file_path}, 原因: {e}")

    print(f"\n✅ 共遍历 status 字段: {total_status_count} 项，其中 Unknown: {unknown_count} 项")

# ==== 使用方法 ====
if __name__ == "__main__":
    json_dir = "/home/lenovo/acm_mm_dataset/instrument_dataset_with_text"  # 替换为你的根路径
    find_all_status_fields(json_dir)
