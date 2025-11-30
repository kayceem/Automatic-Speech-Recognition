import comet_ml
import os 
import argparse
import pytorch_lightning as pl
import torch
import json
from datetime import datetime

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger

from asr.model import ASRTrainer, ConformerASR
from utils import get_models_dir

from dotenv import load_dotenv
load_dotenv()

from asr.dataset import SpeechDataModule

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_float32_matmul_precision('medium')
    
    # Set seed for reproducibility
    pl.seed_everything(args.seed, workers=True)
    if torch.cuda.is_available():
        torch.use_deterministic_algorithms(True, warn_only=True)
    # Build augmentation config from arguments
    augmentation_config = {
        'use_spec_augment': args.use_spec_augment,
        'use_time_mask': args.use_time_mask,
        'use_freq_mask': args.use_freq_mask,
        'use_time_stretch': args.use_time_stretch,
        'use_noise_mixing': args.use_noise_mixing,
        'time_mask_param': args.time_mask_param,
        'time_mask_prob': args.time_mask_prob,
        'time_mask_count': args.time_mask_count,
        'freq_mask_param': args.freq_mask_param,
        'time_stretch_rate': args.time_stretch_rate,
        'noise_snr_db': args.noise_snr_db,
    }
    
    # Prepare dataset
    data_module = SpeechDataModule(
        batch_size=args.batch_size,
        train_json=args.train_json,
        test_json=args.valid_json,
        num_workers=args.num_workers,
        augmentation_config=augmentation_config,
    )
    data_module.setup()

    # Define model hyperparameters
    encoder_params = {
        "d_input": 80,
        "d_model": 144,
        "num_layers": 16,
        "conv_kernel_size": 31,
        "feed_forward_residual_factor": 0.5,
        "feed_forward_expansion_factor": 4,
        "num_heads": 4,
        "dropout": 0.1,
    }

    decoder_params = {
        "d_encoder": 144,
        "d_decoder": 320,
        "num_layers": 1,
        "num_classes": 29,
    }

    # Optimize Model Instance for faster training
    model = ConformerASR(encoder_params, decoder_params)
    model = torch.compile(model)

    speech_trainer = ASRTrainer(model=model, args=args)

    # Build experiment name for logging
    aug_name = []
    if args.use_spec_augment:
        if args.use_time_mask:
            aug_name.append("TM")
        if args.use_freq_mask:
            aug_name.append("FM")
    if args.use_time_stretch:
        aug_name.append("TS")
    if args.use_noise_mixing:
        aug_name.append("NM")
    
    experiment_name = f"seed_{args.seed}_{'_'.join(aug_name) if aug_name else 'no_aug'}"
    
    # NOTE: Comet Logger
    comet_logger = CometLogger(
        api_key=os.getenv("API_KEY"), 
        project_name=os.getenv("PROJECT_NAME"),
        experiment_name=experiment_name,
    )
    
    # Log augmentation config
    comet_logger.log_hyperparams(augmentation_config)
    comet_logger.log_hyperparams({'seed': args.seed})

    # NOTE: Define Trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=get_models_dir("asr") / "checkpoints" / experiment_name,
        filename="Conformer-{epoch:02d}-{val_loss:.2f}-{val_wer:.2f}",
        save_top_k=3,
        mode="min",
    )

    # Trainer Instance
    trainer_args = {
        "accelerator": device,
        "devices": args.gpus,
        "strategy": args.dist_backend if args.gpus > 1 else "auto",
        "min_epochs": 1,
        "max_epochs": args.epochs,
        "precision": args.precision,
        "check_val_every_n_epoch": 1,
        "gradient_clip_val": 1.0,
        "accumulate_grad_batches": 2,
        "callbacks": [
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val_loss", patience=10),
            checkpoint_callback,
        ],
        "logger": comet_logger,

    }

    trainer = pl.Trainer(**trainer_args)

    # Train and Validate
    trainer.fit(speech_trainer, data_module, ckpt_path=args.checkpoint_path)
    val_results = trainer.validate(speech_trainer, data_module)
    
    # Log results locally
    if args.log_local:
        log_results_locally(args, augmentation_config, val_results, experiment_name)
    
    return val_results


def log_results_locally(args, augmentation_config, val_results, experiment_name):
    """Log ablation results to a local JSON file"""
    log_dir = "ablation_logs"
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, "ablation_results.json")
    
    # Prepare result entry
    result_entry = {
        "timestamp": datetime.now().isoformat(),
        "experiment_name": experiment_name,
        "seed": args.seed,
        "augmentation_config": augmentation_config,
        "validation_results": val_results[0] if val_results else {},
        "hyperparameters": {
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "epochs": args.epochs,
        }
    }
    
    # Load existing results or create new list
    if os.path.exists(log_file):
        with open(log_file, 'r') as f:
            results = json.load(f)
    else:
        results = []
    
    results.append(result_entry)
    
    # Save updated results
    with open(log_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n{'='*50}")
    print(f"Results logged locally to: {log_file}")
    print(f"Experiment: {experiment_name}")
    print(f"{'='*50}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASR Model with Ablation Study Support")

    # Train Device Hyperparameters
    parser.add_argument('-g', '--gpus', default=1, type=int, help='number of gpus per node')
    parser.add_argument('-w', '--num_workers', default=8, type=int, help='n data loading workers')
    parser.add_argument('-db', '--dist_backend', default='ddp', type=str,
                        help='which distributed backend to use for aggregating multi-gpu train')

    # Train and Valid File
    parser.add_argument('--train_json', default=None, required=True, type=str, help='json file to load training data')                   
    parser.add_argument('--valid_json', default=None, required=True, type=str, help='json file to load testing data')

    # General Train Hyperparameters
    parser.add_argument('--epochs', default=100, type=int, help='number of total epochs to run')
    parser.add_argument('--batch_size', default=32, type=int, help='size of batch')
    parser.add_argument('-lr', '--learning_rate', default=4e-5, type=float, help='learning rate')
    parser.add_argument('--precision', default='bf16-mixed', type=str, help='precision')
    
    # Checkpoint path
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path to a checkpoint file to resume training')
    
    # Ablation Study: Seed for reproducibility
    parser.add_argument('--seed', default=42, type=int, help='random seed for reproducibility')
    
    # Ablation Study: Augmentation Flags
    parser.add_argument('--use_spec_augment', action='store_true', default=True, 
                        help='use SpecAugment (time + freq masking)')
    parser.add_argument('--no_spec_augment', dest='use_spec_augment', action='store_false',
                        help='disable SpecAugment')
    
    parser.add_argument('--use_time_mask', action='store_true', default=True,
                        help='use time masking')
    parser.add_argument('--no_time_mask', dest='use_time_mask', action='store_false',
                        help='disable time masking')
    
    parser.add_argument('--use_freq_mask', action='store_true', default=True,
                        help='use frequency masking')
    parser.add_argument('--no_freq_mask', dest='use_freq_mask', action='store_false',
                        help='disable frequency masking')
    
    parser.add_argument('--use_time_stretch', action='store_true', default=False,
                        help='use time stretching')
    
    parser.add_argument('--use_noise_mixing', action='store_true', default=False,
                        help='use noise mixing')
    
    # Ablation Study: Augmentation Parameters
    parser.add_argument('--time_mask_param', default=15, type=int, 
                        help='maximum possible length of the time mask')
    parser.add_argument('--time_mask_prob', default=0.05, type=float,
                        help='probability of applying each time mask')
    parser.add_argument('--time_mask_count', default=10, type=int,
                        help='number of time masks to apply')
    parser.add_argument('--freq_mask_param', default=27, type=int,
                        help='maximum possible length of the frequency mask')
    parser.add_argument('--time_stretch_rate', default=1.0, type=float,
                        help='time stretch rate (1.0 = no stretch)')
    parser.add_argument('--noise_snr_db', default=20.0, type=float,
                        help='signal-to-noise ratio in dB for noise mixing')
    
    # Local Logging
    parser.add_argument('--log_local', action='store_true', default=True,
                        help='log results to local JSON file')
    
    args = parser.parse_args()
    main(args)