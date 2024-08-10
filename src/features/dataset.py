import torch
import torchaudio
import numpy as np
import pandas as pd

from torch.utils.data import Dataset, DataLoader, Sampler
from torch.nn.utils.rnn import pad_sequence
from torchaudio.utils import download_asset


CHARS: list = [
    ' ',
    'а',
    'б',
    'в',
    'г',
    'д',
    'е',
    'ж',
    'з',
    'и',
    'й',
    'к',
    'л',
    'м',
    'н',
    'о',
    'п',
    'р',
    'с',
    'т',
    'у',
    'ф',
    'х',
    'ц',
    'ч',
    'ш',
    'щ',
    'ъ',
    'ы',
    'ь',
    'э',
    'ю',
    'я',
]
INT_TO_CHAR = dict(enumerate(CHARS))
CHAR_TO_INT = {v: k for k, v in INT_TO_CHAR.items()}

SPEC_AUG = torch.nn.Sequential(
    # torchaudio.transforms.TimeStretch(0.9, fixed_rate=True),
    torchaudio.transforms.FrequencyMasking(freq_mask_param=20),
    torchaudio.transforms.TimeMasking(time_mask_param=10),
)


class ASRDataset(Dataset):
    def __init__(
        self,
        mode: str,
        audio_files: pd.DataFrame,
        noise_files: list = None,
        spec_aug: torchaudio.transforms = None,
        sr: int = 16_000,
        n_mels: int = 64,
        n_fft: int = 512,
        speech_augment_prop: float = 0.2,
        reverberation_prop: float = 0.02,
    ):
        """
        Датасет для обучения ASR модели. За основу взят датасет Golos от Сбера.
        :param mode: train or not. Влиет на аугментации
        :param audio_files: Датафрейм с указанием путей к аудио (audio_filepath), текстом на записи (text), длительностью записи (duration)
        :param noise_files: Список путей до аудио с шумом
        :param spec_aug: Аугментации спектрограм
        :param sr: sample rate
        :param n_mels: количество mel фильтров
        :param n_fft: размер окна
        :param speech_augment_prop: вероятность наложения шума из noise_files
        :param reverberation_prop: вероятность применения эхо-преобразования
        """
        super(ASRDataset, self).__init__()

        self.mode = mode
        self.collection = audio_files
        self.noise_files = noise_files
        self.sr = sr
        self.n_mels = n_mels
        self.n_fft = n_fft
        self.spec_aug = spec_aug
        self.speech_augment_prop = speech_augment_prop
        self.reverberation_prop = reverberation_prop

        if self.noise_files is None:
            self.speech_augment_prop = 0.0

        self.rir = None

        self.to_melspec = torchaudio.transforms.MelSpectrogram(
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
        self.to_spec = torchaudio.transforms.Spectrogram(
            n_fft=self.n_fft,
            win_length=320,
            hop_length=160,
            window_fn=torch.hann_window,
            power=2,
            wkwargs={"periodic": False},
        )
        self.to_melscale = torchaudio.transforms.MelScale(
            n_mels=self.n_mels,
            sample_rate=self.sr,
            n_stft=self.n_fft // 2 + 1,
            f_min=0.0,
            mel_scale="slaney",
            norm="slaney",
        )
        self.preemphasis = torchaudio.transforms.Preemphasis(coeff=0.97)

    def apply_reverberation(self, waveform: torch.Tensor):
        if self.rir is None:
            torch_rir = download_asset("tutorial-assets/Lab41-SRI-VOiCES-rm1-impulse-mc01-stu-clo-8000hz.wav")
            rir_raw, sample_rate = torchaudio.load(torch_rir)
            rir_raw = torchaudio.transforms.Resample(sample_rate, self.sr)(rir_raw)
            rir = rir_raw[:, int(self.sr * 1.01): int(self.sr * 1.3)]
            self.rir = rir / torch.linalg.vector_norm(rir, ord=2)

        return torchaudio.functional.fftconvolve(waveform, self.rir)

    def apply_background_noise(self, waveform: torch.Tensor):
        noise_file = self.noise_files[
            np.random.randint(0, len(self.noise_files))
        ]
        noise, sample_rate = torchaudio.load(noise_file)
        noise = torchaudio.transforms.Resample(sample_rate, self.sr)(noise)
        noise = noise[noise != 0].unsqueeze(0)
        if noise.shape[1] == 0:
            return waveform

        # повторяем шум для наложения на все аудио
        num_repeats = int(waveform.shape[1] / noise.shape[1] + 1)
        noise = noise.repeat((1, num_repeats))

        # обрезаем лишнюю длину шума
        start_ind_noise = np.random.randint(low=0, high=noise.shape[1] - waveform.shape[1])
        noise = noise[:, start_ind_noise: start_ind_noise + waveform.shape[1]]

        snr_dbs = torch.tensor([np.random.randint(1, 20)])
        noisy_speeches = torchaudio.functional.add_noise(waveform, torch.clamp(noise, min=1e-10), snr_dbs)
        return noisy_speeches[0:1]

    def __getitem__(self, item):
        audio_path = self.collection["audio_filepath"][item]
        transcript = self.collection["text"][item]

        transcript_tensor = torch.tensor(
            [CHAR_TO_INT[i] for i in transcript], dtype=torch.long
        )

        waveform, sample_rate = torchaudio.load(audio_path, normalize=True)
        waveform = torchaudio.transforms.Resample(sample_rate, self.sr)(waveform)

        if self.mode == "train":
            # add random Normal(0, 1) noise
            waveform = waveform + torch.randn_like(waveform) * 1e-5

        # apply preemphasis
        waveform = self.preemphasis(waveform)

        # apply background noise
        if np.random.rand() < self.speech_augment_prop:
            waveform = self.apply_background_noise(waveform)

        # apply room reverberation
        if np.random.rand() < self.reverberation_prop:
            waveform = self.apply_reverberation(waveform)

        # apply spectrogram augmentations
        if self.spec_aug is None:
            mel_spectrogram = self.to_melspec(waveform)
        else:
            # Convert to power spectrogram
            spectrogram = self.to_spec(waveform)
            # Apply SpecAugment
            aug_spectrogram = self.spec_aug(spectrogram)
            # Convert to mel-scale
            mel_spectrogram = self.to_melscale(aug_spectrogram)

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
        return len(self.collection)


def collate_fn(batch):
    melspecs = [row["melspec"] for row in batch]
    texts = [row["text"] for row in batch]
    input_lengths = [row["input_length"] for row in batch]
    label_lengths = [row["label_length"] for row in batch]

    specs = [torch.transpose(spec, 0, 1) for spec in melspecs]
    specs = pad_sequence(specs, batch_first=True)
    specs = torch.transpose(specs, 1, 2)

    labels = pad_sequence(texts, batch_first=True)
    return (
        specs,
        labels,
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
