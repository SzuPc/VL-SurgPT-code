import os
import json

def replace_status_value(obj, target="Self-occlusion", replacement="Self occlusion"):
    for frame, annotations in obj.items():
        for ann in annotations:
            if "status" in ann and isinstance(ann["status"], list):
                ann["status"] = [replacement if s == target else s for s in ann["status"]]
    return obj

def process_all_json_files(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename.endswith(".json"):
                json_path = os.path.join(dirpath, filename)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    updated_data = replace_status_value(data)

                    with open(json_path, 'w') as f:
                        json.dump(updated_data, f, indent=2)

                    print(f"[已修改] {json_path}")
                except Exception as e:
                    print(f"[错误] 无法处理: {json_path}")
                    print(e)

# === 用法 ===
root_dir = "/home/lenovo/acm_mm_dataset/instrument_text_guidance_dataset"  # 替换为目标目录
process_all_json_files(root_dir)
