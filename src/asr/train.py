import comet_ml
import os 
import argparse
import pytorch_lightning as pl
import torch

from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CometLogger

from asr.model import ASRTrainer, ConformerASR
from utils import get_models_dir

from dotenv import load_dotenv
load_dotenv()

from asr.dataset import SpeechDataModule

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Prepare dataset
    data_module = SpeechDataModule(
        batch_size=args.batch_size,
        train_json=args.train_json,
        test_json=args.valid_json,
        num_workers=args.num_workers,
    )
    data_module.setup()

    # Define model hyperparameters
    # https://arxiv.org/pdf/2005.08100 : Table 1 for conformer parameters
    encoder_params = {
        "d_input": 80,          # Input features: n-mels
        "d_model": 144,         # Encoder Dims
        "num_layers": 16,       # Encoder Layers
        "conv_kernel_size": 31,
        "feed_forward_residual_factor": 0.5,
        "feed_forward_expansion_factor": 4,
        "num_heads": 4,         # Relative MultiHead Attetion Heads
        "dropout": 0.1,
    }

    decoder_params = {
        "d_encoder": 144,       # Match with Encoder layer
        "d_decoder": 320,       # Decoder Dim
        "num_layers": 1,        # Deocder Layer
        "num_classes": 29,      # Output Classes
    }

    # Optimize Model Instance for faster training
    model = ConformerASR(encoder_params, decoder_params)
    model = torch.compile(model)

    speech_trainer = ASRTrainer(model=model, args=args)

    # NOTE: Comet Logger
    comet_logger = CometLogger(
        api_key=os.getenv("API_KEY"), project_name=os.getenv("PROJECT_NAME")
    )

    # NOTE: Define Trainer callbacks
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        dirpath=get_models_dir("asr") / "checkpoints",
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
        "accumulate_grad_batches": args.accumulate_grad_batches,
        "callbacks": [
            LearningRateMonitor(logging_interval="epoch"),
            EarlyStopping(monitor="val_loss", patience=5),
            checkpoint_callback,
        ],
        "logger": comet_logger,

    }

    trainer = pl.Trainer(**trainer_args)

    # Train and Validate
    trainer.fit(speech_trainer, data_module, ckpt_path=args.checkpoint_path)
    trainer.validate(speech_trainer, data_module)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train ASR Model")

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
    parser.add_argument('--precision', default='16-mixed', type=str, help='precision')
    
    # Checkpoint path
    parser.add_argument('--checkpoint_path', default=None, type=str, help='path to a checkpoint file to resume training')
    parser.add_argument('--accumulate_grad_batches', default=8, type=int, help='Accumulate gradients over n batches')
    
    args = parser.parse_args()
    main(args)