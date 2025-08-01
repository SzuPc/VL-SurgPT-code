import os
import json

def convert_fields_to_list(obj, keys_to_convert=["location", "instrument_name", "status"]):
    for frame, annotations in obj.items():
        for ann in annotations:
            for key in keys_to_convert:
                if key in ann and not isinstance(ann[key], list):
                    ann[key] = [ann[key]]
    return obj

def process_texts_json_files(root_folder):
    for dirpath, _, filenames in os.walk(root_folder):
        for filename in filenames:
            if filename == "texts.json":  # 只处理这个名字
                json_path = os.path.join(dirpath, filename)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)

                    updated_data = convert_fields_to_list(data)

                    with open(json_path, 'w') as f:
                        json.dump(updated_data, f, indent=2)

                    print(f"[已处理] {json_path}")
                except Exception as e:
                    print(f"[错误] 无法处理文件: {json_path}")
                    print(e)



# === 用法 ===
root_dir = "/home/lenovo/acm_mm_dataset/instrument_text_guidance_dataset"
process_texts_json_files(root_dir)
