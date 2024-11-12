import argparse
import torch
import numpy as np
import pandas as pd
import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint

from src.decoder.greedy_decoder import GreedyCTCDecoder
from src.features.dataset import ASRDataset, BatchSampler, collate_fn, SPEC_AUG, CHARS, INT_TO_CHAR
from src.model.quartznet_torch import QuartzNet
from src.model.quartznet_lightning import QuartzNetLightning, CustomCTCLoss


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--train_manifest', required=True, dest='train_manifest')
    args_parser.add_argument('--valid_manifest', required=True, dest='valid_manifest')
    args_parser.add_argument('--checkpoint_path', default=None, dest='checkpoint_path')
    args_parser.add_argument('--batch_size', default=32, dest='batch_size', type=int)
    args_parser.add_argument('--max_epochs', default=5, dest='max_epochs', type=int)
    args_parser.add_argument('--accumulate_grad_batches', default=64, dest='accumulate_grad_batches', type=int)
    args = args_parser.parse_args()

    # example for train_opus/manifest.jsonl from Golos Dataset
    train_manifest = (
        pd.read_json(path_or_buf=args.train_manifest, lines=True)
    )
    valid_manifest = (
        pd.read_json(path_or_buf=args.valid_manifest, lines=True)
    )

    train_dataset = ASRDataset(mode='train', audio_files=train_manifest, noise_files=None, spec_aug=SPEC_AUG)
    valid_dataset = ASRDataset(mode='valid', audio_files=valid_manifest, noise_files=None, spec_aug=None)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=collate_fn,
        sampler=BatchSampler(np.argsort(train_manifest['duration'].values).tolist()[::-1], args.batch_size),
        num_workers=7,
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=7,
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir='lightning_logs/',
        version='version_0',
    )

    checkpoint_callback = ModelCheckpoint(
        dirpath="./quartznet_ckpts",
        save_last=True,
        save_top_k=-1,
        filename="quartznet-{epoch:02d}-{val_loss:.3f}",
        every_n_epochs=1,
    )

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=logger,
        callbacks=[checkpoint_callback, ],
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision="16-mixed",
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        gradient_clip_val=5,
        use_distributed_sampler=True,
    )

    if args.checkpoint_path:
        # for example: checkpoint_path = 'quartznet_ckpts/last.ckpt'
        model = QuartzNetLightning.load_from_checkpoint(args.checkpoint_path)

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=args.checkpoint_path,
        )

    else:
        pre_trained_model = QuartzNet(
            R_repeat=2,
            in_channels=64,
            out_channels=len(CHARS) + 1,  # for blank token (ctc loss)
        )
        pre_trained_model.load_state_dict(torch.load('quartznet15x5_sber_transfer.state_dict'))

        model = QuartzNetLightning(
            model=pre_trained_model,
            criterion=CustomCTCLoss(blank=len(CHARS)),
            decoder=GreedyCTCDecoder(labels=INT_TO_CHAR, blank=len(CHARS)),
        )

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )
