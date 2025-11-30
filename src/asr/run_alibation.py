#!/usr/bin/env python3
"""
Script to run ablation studies across multiple seeds and configurations.
This automates the process of testing different augmentation combinations.
"""

import subprocess
import argparse
import json
import os
from pathlib import Path

def run_experiment(base_cmd, config_name, config_flags, seeds):
    """Run a single configuration across multiple seeds"""
    print(f"\n{'='*60}")
    print(f"Running Configuration: {config_name}")
    print(f"Flags: {' '.join(config_flags)}")
    print(f"Seeds: {seeds}")
    print(f"{'='*60}\n")
    
    for seed in seeds:
        cmd = base_cmd + config_flags + [f"--seed", str(seed)]
        print(f"Running seed {seed}...")
        print(f"Command: {' '.join(cmd)}\n")
        
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error running experiment with seed {seed}: {e}")
            continue

def main():
    parser = argparse.ArgumentParser(description="Run ablation study for ASR model")
    parser.add_argument('--train_json', required=True, type=str, help='training data json')
    parser.add_argument('--valid_json', required=True, type=str, help='validation data json')
    parser.add_argument('--seeds', nargs='+', type=int, default=[42, 123, 456, 789, 1011],
                        help='seeds to run (default: 5 seeds)')
    parser.add_argument('--batch_size', default=32, type=int, help='batch size')
    parser.add_argument('--epochs', default=40, type=int, help='number of epochs')
    parser.add_argument('--gpus', default=1, type=int, help='number of GPUs')
    parser.add_argument('--quick_test', action='store_true', 
                        help='run quick test with fewer configs and seeds')
    
    args = parser.parse_args()
    
    # Base command for all experiments
    base_cmd = [
        "python", "src/asr/train.py",
        "--train_json", args.train_json,
        "--valid_json", args.valid_json,
        "--batch_size", str(args.batch_size),
        "--epochs", str(args.epochs),
        "--gpus", str(args.gpus),
        "--log_local"
    ]
    
    # Seeds to use
    seeds = args.seeds if not args.quick_test else args.seeds[:2]
    
    # Define ablation configurations
    configs = {
        # "baseline_full_aug": [
        #     "--use_spec_augment",
        #     "--use_time_mask",
        #     "--use_freq_mask"
        # ],
        # "no_augmentation": [
        #     "--no_spec_augment"
        # ],
        # "only_time_mask": [
        #     "--use_spec_augment",
        #     "--use_time_mask",
        #     "--no_freq_mask"
        # ],
        "with_time_stretch": [
            "--no_spec_augment",
            "--use_time_stretch",
            "--time_stretch_rate", "0.9"
        ],
        # "with_noise_mixing": [
        #     "--no_spec_augment",
        #     "--use_noise_mixing",
        #     "--noise_snr_db", "15.0"
        # ],
        "all_augmentations": [
            "--use_spec_augment",
            "--use_time_mask",
            "--use_freq_mask",
            "--use_time_stretch",
            "--time_stretch_rate", "0.9",
            "--use_noise_mixing",
            "--noise_snr_db", "15.0"
        ]
    }
    
    # For quick testing, use fewer configs
    if args.quick_test:
        configs = {k: v for i, (k, v) in enumerate(configs.items()) if i < 3}
    
    # Run all configurations
    for config_name, config_flags in configs.items():
        run_experiment(base_cmd, config_name, config_flags, seeds)
    
    print(f"\n{'='*60}")
    print("Ablation study complete!")
    print("Results saved in: ablation_logs/ablation_results.json")
    print(f"{'='*60}\n")

if __name__ == "__main__":
    main()