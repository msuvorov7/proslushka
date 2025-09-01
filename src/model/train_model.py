import os
import sys
import logging
import argparse
import numpy as np
import pandas as pd

import torch
import pytorch_lightning as pl

import src.decoder.greedy_decoder as greedy_decoder
import src.features.dataset as dataset
import src.model.loss as loss
import src.model.config as config
import src.model.lightning_model as lightning_model

torch.set_num_threads(8)
# torch.set_float32_matmul_precision('medium')

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def prepare_manifest(
    manifest: pd.DataFrame,
    min_audio_duration: float,
    max_audio_duration: float,
) -> pd.DataFrame:
    manifest = manifest[
        (manifest['duration'] < max_audio_duration)
        & (manifest['duration'] > min_audio_duration)
    ].reset_index(drop=True)

    manifest['text'] = manifest['text'].fillna('')
    manifest['duration_int'] = manifest['duration'].apply(int)
    return manifest


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--model', required=True, dest='model')
    args_parser.add_argument('--dataset_path', required=True, dest='dataset_path')
    args_parser.add_argument('--train_manifest', required=True, dest='train_manifest')
    args_parser.add_argument('--valid_manifest', required=True, dest='valid_manifest')
    args_parser.add_argument('--checkpoint_path', default=None, dest='checkpoint_path')
    args_parser.add_argument('--batch_size', default=48, dest='batch_size', type=int)
    args_parser.add_argument('--max_epochs', default=5, dest='max_epochs', type=int)
    args_parser.add_argument('--accumulate_grad_batches', default=64, dest='accumulate_grad_batches', type=int)
    args = args_parser.parse_args()

    assert args.model in ('quartznet', 'citrinet', 'conformer')

    if args.train_manifest.endswith('.jsonl'):
        # example for train_opus/manifest.jsonl from Golos Dataset
        train_manifest = pd.read_json(path_or_buf=args.train_manifest, lines=True)
        valid_manifest = pd.read_json(path_or_buf=args.valid_manifest, lines=True)
    elif args.train_manifest.endswith('.csv'):
        # example for csv
        train_manifest = pd.read_csv(args.train_manifest)
        valid_manifest = pd.read_csv(args.valid_manifest)
    else:
        raise NotImplementedError

    logging.info('manifests loaded')

    train_manifest = prepare_manifest(train_manifest, 0.5, 41)
    valid_manifest = prepare_manifest(valid_manifest, 0.5, 41)
    logging.info(f'manifests prepared: {train_manifest.duration.sum() / 3600}')

    if args.model == 'quartznet':
        pre_trained_sd = 'models/quartznet15x5_sber_transfer.state_dict'
    elif args.model == 'citrinet':
        pre_trained_sd = 'models/citrinet384_10epoch.state_dict'
    elif args.model == 'conformer':
        pre_trained_sd = 'models/conformer_small_176.state_dict'

    pre_trained_model = config.ASRModel(
        model_name=args.model,
        corpus=train_manifest['text'].drop_duplicates().values,
        # pre_trained_tokenizer=f"models/{args.model}_tokenizer.json",
        # pre_trained_sd=pre_trained_sd,
    )
    pre_trained_model.model_tokenizer.save(f"models/{args.model}_tokenizer.json")

    train_dataset = dataset.AudioDataset(
        tokenizer=pre_trained_model.model_tokenizer,
        mode='train',
        audio_files=train_manifest,
        spec_aug=dataset.SPEC_AUG,
        sr=16_000,
        n_mels=pre_trained_model.asr_model['n_mels'],
        n_fft=512,
        normalize=True,
        dataset_path=args.dataset_path,
    )
    valid_dataset = dataset.AudioDataset(
        tokenizer=pre_trained_model.model_tokenizer,
        mode='valid',
        audio_files=valid_manifest,
        spec_aug=None,
        sr=16_000,
        n_mels=pre_trained_model.asr_model['n_mels'],
        n_fft=512,
        normalize=True,
        dataset_path=args.dataset_path,
    )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        collate_fn=dataset.collate_fn,
        sampler=dataset.BatchSampler(np.argsort(train_manifest['duration_int'].values).tolist()[::-1], args.batch_size),
        # num_workers=8,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=dataset.collate_fn,
        # num_workers=8,
    )

    logger = pl.loggers.TensorBoardLogger(
        save_dir='lightning_logs/',
        version='version_0',
    )

    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath="./ckpts",
        save_last=True,
        save_top_k=-1,
        filename="{args.model}-{epoch:02d}-{val_loss:.3f}",
        every_n_epochs=1,
    )
    lr_callback =pl.callbacks.LearningRateMonitor(
        logging_interval='step',
    )

    trainer = pl.Trainer(
        accelerator='gpu' if torch.cuda.is_available() else 'cpu',
        logger=logger,
        callbacks=[checkpoint_callback, lr_callback],
        max_epochs=args.max_epochs,
        accumulate_grad_batches=args.accumulate_grad_batches,
        precision="bf16-mixed",
        log_every_n_steps=1,
        enable_checkpointing=True,
        enable_progress_bar=True,
        gradient_clip_val=5,
        use_distributed_sampler=True,
    )

    if args.checkpoint_path:
        # for example: checkpoint_path = 'quartznet_ckpts/last.ckpt'
        model = lightning_model.ASRLightning.load_from_checkpoint(args.checkpoint_path)

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
            ckpt_path=args.checkpoint_path,
        )

    else:
        model = lightning_model.ASRLightning(
            model=pre_trained_model.asr_model['model'],
            criterion=loss.CTCLoss(pre_trained_model.model_tokenizer.get_vocab_size()),
            decoder=greedy_decoder.GreedyCTCDecoder(
                tokenizer=pre_trained_model.model_tokenizer,
                blank=pre_trained_model.model_tokenizer.get_vocab_size()
            ),
            t_max=int(len(train_dataset) / (args.batch_size * args.accumulate_grad_batches) + 1) * args.max_epochs,
            inputs_length_scale=pre_trained_model.asr_model['input_scale'],
            lr=0.05,
        )

        trainer.fit(
            model,
            train_dataloaders=train_loader,
            val_dataloaders=valid_loader,
        )

    # save state dict
    torch.save(model.model.state_dict(), f'models/{args.model}_{args.max_epochs}_epoch.state_dict')
    logging.info('state dict saved in models/')

    # save in onnx format
    dummy_input = torch.randn(1, pre_trained_model.asr_model['n_mels'], 256, dtype=torch.float32)
    torch.onnx.export(
        model.model.eval().to('cpu'),
        dummy_input,
        f'models/{args.model}_{args.max_epochs}_epoch.onnx',
        input_names=['mel_spec'],
        dynamic_axes={'mel_spec': {0: 'batch_size', 2: 'time'}}
    )
    logging.info('onnx model saved in models/')