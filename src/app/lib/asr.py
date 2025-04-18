import numpy as np
import soundfile as sf

from itertools import groupby

import lib.librosa as librosa
import lib.vad as vad


def log_softmax(x, axis=None):
    x_max = np.amax(x, axis=axis, keepdims=True)

    if x_max.ndim > 0:
        x_max[~np.isfinite(x_max)] = 0
    elif not np.isfinite(x_max):
        x_max = 0

    tmp = x - x_max
    exp_tmp = np.exp(tmp)

    # suppress warnings about log of zero
    with np.errstate(divide='ignore'):
        s = np.sum(exp_tmp, axis=axis, keepdims=True)
        out = np.log(s)

    return tmp - out


def split_audio_on_batch(labels: list[dict], min_duration: float, audio_len: int) -> list[dict]:
    batches = [
        {
            'speech_begin_ids': 0,
            'speech_begin': 0,
            'segment_duration': 0,
        }
    ]

    for seg in labels:
        if batches[-1]['segment_duration'] < min_duration:
            batches[-1]['segment_duration'] = seg['speech_end'] - batches[-1]['speech_begin']
            batches[-1]['speech_end'] = seg['speech_end']
            batches[-1]['speech_end_ids'] = seg['speech_end_ids']
        else:
            batches.append({
                'speech_begin_ids': batches[-1]['speech_end_ids'],
                'speech_begin': batches[-1]['speech_end'],
                'segment_duration': 0,
            })
    # pack audio ending with last batch
    batches[-1]['speech_end_ids'] = audio_len
    return batches


def audio_transcribe(audio: np.ndarray, sample_rate: int, model, tokenizer) -> str:
    mel_spec = librosa.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=512,
        hop_length=160,
        win_length=320,
        window='hann',
        pad_mode='reflect',
        power=2,
        fmin=0,
        n_mels=80,
        norm='slaney',
        center=True
    )

    model_input = {
        model.get_inputs()[0].name: np.log(mel_spec)[np.newaxis, :, :].astype(np.float32)
    }
    model_output = model.run(None, model_input)[0]
    output = model_output.transpose(2, 0, 1)
    output = log_softmax(output, axis=2)
    output = output.transpose(1, 0, 2)

    tokens = output.argmax(axis=2).tolist()[0]
    # drop duplicated tokens
    tokens = [key for key, _group in groupby(tokens)]

    return tokenizer.decode(tokens).replace(' ##', '')


def speech_to_text(
    file_path: str,
    onnx_model,
    tokenizer,
    target_sr: int = 16_000,
    max_audio_duration: float = 30,
    min_batch_audio_duration: float = 15,
) -> str:
    wav, sr = sf.read(file_path)
    wav = librosa.to_mono(wav)
    wav = librosa.resample(wav, orig_sr=sr, target_sr=target_sr)
    wav = librosa.preemphasis(wav, coef=0.97)

    texts = []

    if (len(wav) // target_sr) > max_audio_duration:
        v = vad.VoiceActivityDetector(wav, target_sr)
        raw_detection = v.detect_speech()
        speech_labels = v.convert_windows_to_readible_labels(raw_detection)

        wav_batches = split_audio_on_batch(speech_labels, min_batch_audio_duration, len(wav))

        for batch in wav_batches:
            texts.append(
                audio_transcribe(
                    wav[batch['speech_begin_ids']: batch['speech_end_ids']],
                    target_sr,
                    onnx_model,
                    tokenizer,
                )
            )
    else:
        texts.append(
            audio_transcribe(
                wav,
                target_sr,
                onnx_model,
                tokenizer,
            )
        )

    return ' '.join(texts)
