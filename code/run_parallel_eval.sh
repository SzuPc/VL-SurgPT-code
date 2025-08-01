#!/bin/bash

SCRIPT_PATH="/home/lenovo/acm_mm_dataset/track_on/eval_tissue_acmmm.py"
PYTHON_BIN="/home/lenovo/anaconda3/envs/trackon/bin/python"  # 可替换成你当前环境的 python 路径

# index: 0, 1 → device 0
for idx in 0 1 2; do
  $PYTHON_BIN $SCRIPT_PATH --index $idx --device 0 &
  sleep 2
done

# index: 2, 3, 4 → device 1
for idx in 3 4; do
  $PYTHON_BIN $SCRIPT_PATH --index $idx --device 1 &
  sleep 2
done

wait  # 等待所有并行任务完成

