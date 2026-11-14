import torch
import torchaudio
import torch.nn as nn


class AudioPreprocessor(nn.Module):
    """
    Выделенный модуль предобработки. 
    """
    def __init__(
        self,
        sr: int = 16_000,
        n_mels: int = 80,
        n_fft: int = 512,
        normalize: bool = True,
    ):
        super().__init__()
        self.sr = sr
        self.n_mels = n_mels
        self.normalize = normalize
        
        self.preemphasis = torchaudio.transforms.Preemphasis(coeff=0.97)
        self.melspec = torchaudio.transforms.MelSpectrogram(
            sample_rate=sr,
            win_length=320,
            hop_length=160,
            n_mels=n_mels,
            window_fn=torch.hann_window,
            mel_scale="slaney",
            norm="slaney",
            n_fft=n_fft,
            f_min=0,
            wkwargs={"periodic": False},
        )

    def forward(self, waveform):
        # 1. Preemphasis
        x = self.preemphasis(waveform)
        # 2. Mel Spectrogram
        x = self.melspec(x)
        # 3. Log (с защитой от минус бесконечности)
        x = torch.log(torch.clamp(x, min=1e-10))
        # 4. Normalization (Per-feature)
        if self.normalize:
            mean = x.mean(dim=-1, keepdim=True)
            std = x.std(dim=-1, keepdim=True)
            x = (x - mean) / (std + 1e-8)
        return x
