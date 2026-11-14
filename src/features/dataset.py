import os
import random

import torch
import torchaudio
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence

from src.audio_preprocessor.audio_preprocessor import AudioPreprocessor


SPEC_AUG = torch.nn.Sequential(
    # torchaudio.transforms.TimeStretch(0.9, fixed_rate=True),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
    torchaudio.transforms.TimeMasking(time_mask_param=10),
)

class AudioDataset(Dataset):
    def __init__(
        self,
        tokenizer,
        mode: str,
        audio_files: pd.DataFrame,
        spec_aug: torch.nn.Module = None,
        sr: int = 16_000,
        n_mels: int = 80,
        dataset_path='../data/'
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.mode = mode
        self.audio_files = audio_files
        self.sr = sr
        self.dataset_path = dataset_path
        self.spec_aug = spec_aug

        self.preprocessor = AudioPreprocessor(
            sr=sr,
            n_mels=n_mels,
            normalize=True,
        )
        
        # Кэш для ресэмплеров (чтобы не пересоздавать фильтры каждый раз)
        self.resamplers = {}

    def _get_resampler(self, orig_freq):
        if orig_freq not in self.resamplers:
            self.resamplers[orig_freq] = torchaudio.transforms.Resample(orig_freq, self.sr)
        return self.resamplers[orig_freq]

    def __getitem__(self, item: int) -> dict:
        audio_path = self.audio_files["audio_filepath"][item]
        transcript = self.audio_files["text"][item]
        
        try:
            full_path = os.path.join(self.dataset_path, audio_path)
            waveform, sample_rate = torchaudio.load(full_path, normalize=True)
            
            # Быстрый ресэмплинг
            if sample_rate != self.sr:
                waveform = self._get_resampler(sample_rate)(waveform)

            # Получаем нормализованную лог-мел спектрограмму
            log_mel_spec = self.preprocessor(waveform)

            # Применяем SpecAugment только в режиме train и ПОСЛЕ нормализации
            if self.mode == "train" and self.spec_aug is not None:
                log_mel_spec = self.spec_aug(log_mel_spec)

            transcript_tensor = torch.tensor(self.tokenizer.encode(transcript).ids, dtype=torch.int64)
            
            return {
                "melspec": log_mel_spec.squeeze(0),
                "text": transcript_tensor,
                "input_length": log_mel_spec.shape[-1],
                "label_length": len(transcript_tensor),
                "is_valid": True,
            }

        except Exception as e:
            # Возвращаем заглушку, чтобы не ронять DataLoader
            return {
                "melspec": torch.zeros(self.preprocessor.n_mels, 1, dtype=torch.float32),
                "text": torch.zeros(0, dtype=torch.int64),
                "input_length": 1,
                "label_length": 0,
                "is_valid": False # По этому флагу мы выкинем мусор в collate_fn
            }

    def __len__(self):
        return len(self.audio_files)


def collate_fn(batch):
    # Фильтруем битые файлы, отловленные в __getitem__
    valid_batch = [b for b in batch if b["is_valid"]]
    
    # Если весь батч оказался битым (крайне редкий случай)
    if len(valid_batch) == 0:
        return None 

    melspecs = []
    texts = []
    input_lengths = []
    label_lengths = []
    
    for row in valid_batch:
        # Транспонируем для pad_sequence (нужно чтобы время было первым измерением)
        melspecs.append(row["melspec"].transpose(0, 1))
        texts.append(row["text"])
        input_lengths.append(row["input_length"])
        label_lengths.append(row["label_length"])

    # Паддинг спектрограмм
    melspecs = pad_sequence(melspecs, batch_first=True)
    # Возвращаем размерность обратно: (batch, n_mels, time)
    melspecs = melspecs.transpose(1, 2)

    # Паддинг текстов
    texts = pad_sequence(texts, batch_first=True)
    
    return (
        melspecs,
        texts,
        torch.tensor(input_lengths, dtype=torch.long),
        torch.tensor(label_lengths, dtype=torch.long),
    )


class FastBatchSampler(Sampler):
    """
    O(1) Sampler. Группирует отсортированные по длине индексы в батчи 
    и перемешивает порядок самих батчей.
    """
    def __init__(self, sorted_inds: list, batch_size: int):
        self.sorted_inds = sorted_inds
        self.batch_size = batch_size
        
        # Заранее нарезаем индексы на чанки
        self.batches = [
            self.sorted_inds[i : i + self.batch_size] 
            for i in range(0, len(self.sorted_inds), self.batch_size)
        ]

    def __iter__(self):
        # Перемешиваем батчи, чтобы эпохи отличались
        random.shuffle(self.batches)
        for batch in self.batches:
            yield from batch

    def __len__(self):
        return len(self.sorted_inds)
