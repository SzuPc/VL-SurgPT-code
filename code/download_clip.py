
from transformers import CLIPTokenizer, CLIPTextModel

model_name = "openai/clip-vit-base-patch32"

# 下载 tokenizer 和 text encoder
CLIPTokenizer.from_pretrained(model_name, cache_dir="./clip_pretrained")
CLIPTextModel.from_pretrained(model_name, cache_dir="./clip_pretrained")
