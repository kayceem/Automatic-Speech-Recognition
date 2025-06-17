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
	sudo apt update
	sudo apt install -y $(SYSTEM_PACKAGES)

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
