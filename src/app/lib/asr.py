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
            encoder,
            encoder_tokenizer,
            target_sr: int = 16_000,
    ):
        self.encoder = encoder
        self.encoder_tokenizer = encoder_tokenizer
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
    
    def loader(self, audio, timestamps, batch_size: int):
        for i in range(0, len(timestamps), batch_size):
            batch = pad_sequence(
                [audio[ts['begin_ids']: ts['end_ids']] for ts in timestamps[i: i + batch_size]]
            )
            yield batch

    def encoder_forward(self, batch: np.ndarray):
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
            self.encoder.get_inputs()[0].name: log_mel_spec
        }
        model_output = self.encoder.run(None, model_input)[0]
        output = model_output.transpose(2, 0, 1)
        output = log_softmax(output, axis=2)
        output = output.transpose(1, 0, 2)

        tokens = output.argmax(axis=2).tolist()

        # drop duplicated tokens
        tokens = [[key for key, _ in groupby(row)] for row in tokens]

        return [_.replace(' ##', '') for _ in self.encoder_tokenizer.decode_batch(tokens)]
    
    def decoder_forward(self, batch: list):
        """
        rut5-spell-correct prototype from UrukHan/t5-russian-spell
        """
        # print(batch)
        inputs = self.decoder_tokenizer.encode_batch([f'Spell correct: {text}' for text in batch])

        input_ids = pad_sequence([_.ids for _ in inputs]).astype(np.int64)  # (1, seq_len)
        attention_mask = pad_sequence([_.attention_mask for _ in inputs]).astype(np.int64)  # (1, seq_len)

        # print(pad_sequence(input_ids))

        # Запуск энкодера
        encoder_outputs = self.decoder[0].run(
            None,
            {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
            }
        )

        # Получение encoder_hidden_states
        encoder_hidden_states = encoder_outputs[0]
        encoder_attention_mask = attention_mask  # можно использовать ту же маску

        # Инициализация декодера
        decoder_input_ids = np.zeros((len(encoder_hidden_states), 1), dtype=np.int64)  # начать с pad
        # или начать с <s> (начала предложения)

        # Генерация
        max_length = 256
        batch_size = len(encoder_hidden_states)
        output_ids = [[] for _ in range(batch_size)]
        finished = [False] * batch_size

        for _ in range(max_length):
            decoder_outputs = self.decoder[1].run(
                None,
                {
                    "input_ids": decoder_input_ids,
                    "encoder_attention_mask": encoder_attention_mask,
                    "encoder_hidden_states": encoder_hidden_states,
                }
            )

            # decoder_outputs[0] — это logits для следующего токена
            next_token_logits = decoder_outputs[0][:, -1, :]
            next_token_id = np.argmax(next_token_logits, axis=-1).reshape(-1, 1)
            decoder_input_ids = np.concatenate([decoder_input_ids, next_token_id], axis=-1)
            
            # Проверяем, достигнут ли eos_token_id
            for i in range(batch_size):
                if not finished[i]:
                    token_id = next_token_id[i, 0]
                    output_ids[i].append(token_id)
                    if token_id == self.decoder_tokenizer.token_to_id('</s>'):
                        finished[i] = True

            # Если все последовательности завершены — выходим
            if all(finished):
                break

        # Декодирование
        corrected_text = self.decoder_tokenizer.decode_batch(output_ids, skip_special_tokens=True)
        return ' '.join(corrected_text)

    def speech_to_text(self, audio: np.ndarray, sample_rate: int) -> str:
        batch_size = 8
        acoustic_output = []
        audio = self.preprocess(audio, sample_rate)
        timestamps = self.split_audio(audio, 30, 5)

        for batch in self.loader(audio, timestamps, batch_size):
            encoder_output = self.encoder_forward(batch)
            acoustic_output += encoder_output

        return ' '.join(acoustic_output)
