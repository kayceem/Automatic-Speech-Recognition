# Makefile for ASR Pipeline

.PHONY: all setup train preprocess decompress compress

SYSTEM_PACKAGES = \
  libasound-dev \
  portaudio19-dev \
  libportaudio2 \
  libportaudiocpp0 \
  ffmpeg \
  sox \
  libsox-fmt-all \
  build-essential \
  zlib1g-dev \
  libbz2-dev \
  liblzma-dev \
  btop \
  nano

# Directories and paths
TRAIN_JSON = ./data/processed/asr/train.json
VALID_JSON = ./data/processed/asr/test.json
VALIDATED_TSV = ./data/raw/common_voice/cv-corpus-11.0-2023-06-21/en/validated.tsv
PROCESSED_DIR = ./data/processed/asr/
ARCHIVE_FILE = common_voice.tar.zst

# ========================
# Main Targets
# ========================

all: train

setup:
	apt update
	apt install -y $(SYSTEM_PACKAGES)

train:
	python -m src.asr.train --train_json $(TRAIN_JSON) --valid_json $(VALID_JSON)

preprocess:
	python -m src.asr.scripts.common_voice \
		--file_path $(VALIDATED_TSV) \
		--save_json_path $(PROCESSED_DIR) \
		--percent 20 \
		-w 110

decompress:
	python compressor.py decompress \
		--archive_path $(ARCHIVE_FILE) \
		--output_dir $(PROCESSED_DIR) \
		--threads 100

compress:
	python compressor.py compress \
		--input_dir $(PROCESSED_DIR) \
		--output_file $(ARCHIVE_FILE) \
		--threads 100 \
		--compression_level 5

jupyter:
	nohup jupyter-lab --allow-root --no-browser --port=8888 --ip=0.0.0.0 \
	  --FileContentsManager.delete_to_trash=False \
	  --ServerApp.terminado_settings='{"shell_command": ["/bin/bash"]}' \
	  --ServerApp.token=ihs3rycmswo4lge2tfwz \
	  --ServerApp.allow_origin='*' \
	  --ServerApp.preferred_dir=/workspace/ASR-with-Speech-Sentiment-and-Text-Summarizer/ \
	  > /dev/null 2>&1 &
