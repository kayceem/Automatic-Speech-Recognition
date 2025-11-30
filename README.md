# Automatic Speech Recognition

## Overview

This project integrates three AI components: **Automatic Speech Recognition (ASR)**, **Speech Emotion Recognition (SER)**, and **Text Summarization**. The system aims to handle diverse accents in noisy environments, interpret emotional tone from speech, and generate meaningful summaries while preserving critical information. Target applications include customer service, business meetings, media, and education.

## Models & Data

The implementation uses **Conformer** for ASR, a **XGBoost** for SER, and **BART Large** for text summarization. Training data comes from Mozilla Common Voice and LibriSpeech for ASR, RAVDESS/CremaD/TESS/SAVEE for emotion recognition, and XSum/CNN Daily for summarization. The project is still in progress, with the text summarizer being the most complete component.

---

## Setup

```bash
# Clone repository with submodules
git clone https://github.com/kayceem/Automatic-Speech-Recognition.git

# Install system dependencies (Linux)
sudo apt update
sudo apt install sox libsox-fmt-all build-essential zlib1g-dev libbz2-dev liblzma-dev

# Install Python dependencies
uv sync -r requirements.txt
```