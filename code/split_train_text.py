import os
import shutil
import random
from pathlib import Path
from tqdm import tqdm

def split_dataset(original_root, output_root, split_ratio=0.8, seed=42):
    random.seed(seed)
    folder_names = ['0', '1', '2', '3', '4']

    for folder in folder_names:
        source_folder = os.path.join(original_root, folder)
        if not os.path.exists(source_folder):
            print(f"Skipping missing folder: {source_folder}")
            continue

        subfolders = sorted(os.listdir(os.path.join(source_folder, 'left')))
        random.shuffle(subfolders)

        split_idx = int(len(subfolders) * split_ratio)
        train_set = subfolders[:split_idx]
        test_set = subfolders[split_idx:]

        for split_name, split_list in [('train', train_set), ('test', test_set)]:
            for seq in split_list:
                src_left = os.path.join(source_folder, 'left', seq)
                src_seg = os.path.join(source_folder, 'left', seq.replace("frames", "segmentation"))
                
                # 复制 left
                dst_left = os.path.join(output_root, split_name, folder, 'left', seq)
                if os.path.exists(src_left):
                    shutil.copytree(src_left, dst_left, dirs_exist_ok=True)

                # 复制 segmentation
                src_seg = os.path.join(source_folder, 'left', seq, 'segmentation')
                if os.path.exists(src_seg):
                    dst_seg = os.path.join(output_root, split_name, folder, 'left', seq, 'segmentation')
                    shutil.copytree(src_seg, dst_seg, dirs_exist_ok=True)

    print(f"Split completed and saved under: {output_root}")

# 示例用法
original_root = '/home/lenovo/acm_mm_dataset/instrument_dataset_with_text'   # 原始包含0~4文件夹的路径
output_root = '/home/lenovo/acm_mm_dataset/instrument_text_guidance_dataset'      # 输出路径，会创建 train/ 和 test/
split_dataset(original_root, output_root)
