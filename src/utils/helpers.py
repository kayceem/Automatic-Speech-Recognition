import os
from pathlib import Path

def get_project_root() -> Path:
    return Path(__file__).resolve().parents[2]

def get_data_dir(subfolder=""):
    path = Path.joinpath(get_project_root(), "data", subfolder)
    os.makedirs(path, exist_ok=True)
    return path

def get_assets_dir(subfolder=""):
    path = Path.joinpath(get_project_root(), "assets", subfolder)
    os.makedirs(path, exist_ok=True)
    return path

def get_models_dir(subfolder=""):
    path = Path.joinpath(get_project_root(), "models", subfolder)
    os.makedirs(path, exist_ok=True)
    return path

def get_processed_data_dir(subfolder=""):
    path = Path.joinpath(get_project_root(), "data", "processed", subfolder)
    os.makedirs(path, exist_ok=True)
    return path