from asr.helpers import TextTransform
from utils import get_assets_dir

text_transform = TextTransform()
index_map = text_transform.index_map

max_index = max(index_map.keys())
tokens = [index_map[i] for i in range(max_index + 1)]

tokens = [" " if token == "<SPACE>" else token for token in tokens]

with open(get_assets_dir() /"tokens.txt", "w") as f:
    for token in tokens:
        f.write(f"{token}\n")

print("tokens.txt generated successfully.")
