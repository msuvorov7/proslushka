import numpy as np

from itertools import groupby

import src.app.lib.librosa as librosa
import src.app.lib.webrtcvad as wervad


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


def pad_sequence(batch: list):
    max_len = max([len(row) for row in batch])
    padded = []
    for row in batch:
        to_pad = max_len - len(row)
        padded.append(np.concatenate([row, np.zeros(to_pad)]))
    return np.vstack(padded)


class ASRModel:
    def __init__(
            self,
            asr_encoder,
            asr_tokenizer,
            punct_model,
            punct_tokenizer,
            target_sr: int = 16_000,
    ):
        self.asr_encoder = asr_encoder
        self.asr_tokenizer = asr_tokenizer
        self.punct_model = punct_model
        self.punct_tokenizer = punct_tokenizer
        self.punct_tokenizer.enable_padding()
        self.target_sr = target_sr

    def preprocess(self, wav: np.ndarray, sr: int) -> np.ndarray:
        wav = librosa.to_mono(wav)
        wav = librosa.resample(wav, orig_sr=sr, target_sr=self.target_sr)
        wav = librosa.preemphasis(wav, coef=0.97)
        return wav
    
    def split_audio(self, audio: np.ndarray, max_duration: int, min_timestamp: int) -> list:
        audio_duration = len(audio) // self.target_sr
        # split long audio with VAD
        if audio_duration > max_duration:
            v = wervad.VoiceActivityDetector(audio, self.target_sr)
            raw_detection = v.detect_speech()
            speech_labels = v.convert_windows_to_readable_labels(raw_detection)

            timestamps = [{
                'begin_ids': 0,
                'end_ids': 0,
                'begin_time': 0,
                'end_time': 0,
                'duration': 0,
            }]

            for segment in speech_labels:
                if timestamps[-1]['duration'] < min_timestamp:
                    timestamps[-1]['duration'] = segment['speech_end'] - timestamps[-1]['begin_time']
                    timestamps[-1]['end_time'] = segment['speech_end']
                    timestamps[-1]['end_ids'] = segment['speech_end_ids']
                else:
                    timestamps.append({
                        'begin_ids': timestamps[-1]['end_ids'],
                        'begin_time': timestamps[-1]['end_time'],
                        'duration': 0,
                    })
            # pack audio ending with last batch
            timestamps[-1]['end_ids'] = len(audio)

        else:
            timestamps = [{
                'begin_ids': 0,
                'end_ids': len(audio),
                'begin_time': 0,
                'end_time': audio_duration,
                'duration': audio_duration,
            }]

        return timestamps
    
    def asr_batch_loader(self, audio, timestamps, batch_size: int):
        for i in range(0, len(timestamps), batch_size):
            batch = pad_sequence(
                [audio[ts['begin_ids']: ts['end_ids']] for ts in timestamps[i: i + batch_size]]
            )
            yield batch

    def asr_encoder_forward(self, batch: np.ndarray):
        mel_spec = librosa.melspectrogram(
            y=batch,
            sr=self.target_sr,
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

        log_mel_spec = np.log(np.clip(mel_spec, a_min=1e-10, a_max=np.inf)).astype(np.float32)
        model_input = {
            self.asr_encoder.get_inputs()[0].name: log_mel_spec
        }
        model_output = self.asr_encoder.run(None, model_input)[0]
        output = model_output.transpose(2, 0, 1)
        output = log_softmax(output, axis=2)
        output = output.transpose(1, 0, 2)

        tokens = output.argmax(axis=2).tolist()

        # drop duplicated tokens
        tokens = [[key for key, _ in groupby(row)] for row in tokens]
        return [_.replace(' ##', '') for _ in self.asr_tokenizer.decode_batch(tokens, skip_special_tokens=True)]
    
    def punct_loader(self, text: str, max_length: int, batch_size: int):
        words = text.split()
        data = []
        for i in range(0, len(words), max_length):
            data.append(' '.join(words[i: i + max_length]))
        for i in range(0, len(data), batch_size):
            yield data[i: i + batch_size]
    
    def punct_forward(self, text: str) -> str:
        batch_size = 8
        max_words = 80
        IND_TO_TARGET_TOKEN = {1: 20, 2: 24, 3: 45, 4: 107, 5: 39, 6: 67}

        result = []
        a = []
        for batch in self.punct_loader(text, max_words, batch_size):
            inputs = self.punct_tokenizer.encode_batch(
                batch,
                add_special_tokens=True,
            )
            input_ids = np.asarray([_.ids for _ in inputs]).astype(np.int64).reshape(len(batch), -1)
            attention_mask = np.asarray([_.attention_mask for _ in inputs]).astype(np.int64).reshape(len(batch), -1)

            model_input = {
                self.punct_model.get_inputs()[0].name: input_ids,
                self.punct_model.get_inputs()[1].name: attention_mask,
            }
            model_output = self.punct_model.run(None, model_input)[0].argmax(axis=-1).reshape(len(batch), -1)

            decoded_batch = []
            for i in range(len(batch)):
                sentence_tokens = []
                for tok, trg in zip(input_ids[i], model_output[i]):
                    sentence_tokens.append(tok)
                    if trg != 0:
                        sentence_tokens.append(IND_TO_TARGET_TOKEN[trg])
                decoded_batch.append(sentence_tokens)

            result += self.punct_tokenizer.decode_batch(decoded_batch, skip_special_tokens=True)

        # postprocessing
        for s in ' '.join(result).split('.'):
            a.append(
                (s.replace(' _ ', '-').strip().strip(',-') + '.').capitalize()
            )

        return ' '.join(a)

    def speech_to_text(self, audio: np.ndarray, sample_rate: int) -> str:
        batch_size = 8
        acoustic_output = []
        audio = self.preprocess(audio, sample_rate)
        timestamps = self.split_audio(audio, 30, 5)

        for batch in self.asr_batch_loader(audio, timestamps, batch_size):
            encoder_output = self.asr_encoder_forward(batch)
            acoustic_output += encoder_output

        return self.punct_forward(' '.join(acoustic_output))
