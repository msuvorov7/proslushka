import os
import torch
import torchaudio
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, Sampler
from torch.nn.utils.rnn import pad_sequence


SPEC_AUG = torch.nn.Sequential(
    # torchaudio.transforms.TimeStretch(0.9, fixed_rate=True),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
    torchaudio.transforms.TimeMasking(time_mask_param=10),
)


class AudioDataset(Dataset):
    """
    Simple version
    """
    def __init__(
        self,
        tokenizer,
        mode: str,
        audio_files: pd.DataFrame,
        spec_aug: torchaudio.transforms = None,
        sr: int = 16_000,
        n_mels: int = 64,
        n_fft: int = 512,
        dataset_path = '../data/'
    ):
        super().__init__()
        self.tokenizer = tokenizer
        self.mode = mode
        self.audio_files = audio_files
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.spec_aug = spec_aug
        self.dataset_path = dataset_path

        self.speed_perturb = torchaudio.transforms.SpeedPerturbation(
            self.sr,
            [0.9, 1.1, 1.0, 1.0, 1.0],
        )
        self.spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=320,
            hop_length=160,
            window_fn=torch.hann_window,
            power=2,
            wkwargs={"periodic": False},
        )
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=self.sr,
            win_length=320,
            hop_length=160,
            n_mels=self.n_mels,
            window_fn=torch.hann_window,
            mel_scale="slaney",
            norm="slaney",
            n_fft=self.n_fft,
            f_max=None,
            f_min=0,
            wkwargs={"periodic": False},
        )
        self.melscale = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sr,
            n_stft=self.n_fft // 2 + 1,
            f_min=0.0,
            mel_scale="slaney",
            norm="slaney",
        )
        self.preemphasis = torchaudio.transforms.Preemphasis(coeff=0.97)

    def __getitem__(self, item: int) -> dict:
        audio_path = self.audio_files["audio_filepath"][item]
        transcript = self.audio_files["text"][item]
        transcript_tensor = torch.tensor(self.tokenizer.encode(transcript).ids, dtype=torch.int64)

        try:
            waveform, sample_rate = torchaudio.load(os.path.join(self.dataset_path, audio_path), normalize=True)
            waveform = torchaudio.transforms.Resample(sample_rate, self.sr)(waveform)

            if self.mode == "train":
                # Applies the speed perturbation augmentation
                waveform, _ = self.speed_perturb(waveform)

            waveform = self.preemphasis(waveform)

            # apply spectrogram augmentations
            if self.spec_aug is None:
                mel_spectrogram = self.melspec(waveform)
            else:
                # Convert to power spectrogram
                spectrogram = self.spec(waveform)
                # Apply SpecAugment
                aug_spectrogram = self.spec_aug(spectrogram)
                # Convert to mel-scale
                mel_spectrogram = self.melscale(aug_spectrogram)
        except:
            with open('broken.txt', 'a') as file:
                file.write(audio_path)
                file.write('\n')

            mel_spectrogram = torch.zeros(1, self.n_mels, 100, dtype=torch.float32)
            transcript_tensor = torch.zeros(5, dtype=torch.int64)

        log_mel_spec = torch.log(torch.clamp(mel_spectrogram, min=1e-10))

        input_length = log_mel_spec.shape[-1]
        label_length = len(transcript_tensor)

        return {
            "melspec": log_mel_spec.squeeze(0),
            "text": transcript_tensor,
            "input_length": input_length,
            "label_length": label_length,
        }

    def __len__(self):
        return len(self.audio_files)


def collate_fn(batch):
    melspecs = []
    texts = []
    input_lengths = []
    label_lengths = []
    for row in batch:
        melspecs.append(torch.transpose(row["melspec"], 0, 1))
        texts.append(row["text"])
        input_lengths.append(row["input_length"])
        label_lengths.append(row["label_length"])

    melspecs = pad_sequence(melspecs, batch_first=True)
    melspecs = torch.transpose(melspecs, 1, 2)

    texts = pad_sequence(texts, batch_first=True)
    return (
        melspecs,
        texts,
        torch.tensor(input_lengths),
        torch.tensor(label_lengths),
    )


class BatchSampler(Sampler):
    """
    Sample по длине аудио
    """
    def __init__(self, sorted_inds: list, batch_size: int):
        self.sorted_inds = sorted_inds
        self.batch_size = batch_size

    def __iter__(self):
        inds = self.sorted_inds.copy()
        while len(inds):
            to_take = min(self.batch_size, len(inds))
            start_ind = np.random.randint(low=0, high=len(inds) - to_take + 1)
            batch_inds = inds[start_ind:start_ind + to_take]
            del inds[start_ind:start_ind + to_take]
            for ind in batch_inds:
                yield ind

    def __len__(self):
        return len(self.sorted_inds)
