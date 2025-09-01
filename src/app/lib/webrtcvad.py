import numpy as np
from scipy import signal
import webrtcvad


class VoiceActivityDetector:
    def __init__(self, audio: np.ndarray, sample_rate: int):
        self.audio = audio
        self.sample_rate = sample_rate
        self.vad = webrtcvad.Vad()
        # Установка агрессивности VAD (0-3, где 3 наиболее агрессивный)
        self.vad.set_mode(1)
        
    def detect_speech(self):
        """Обнаружение речи с использованием WebRTC VAD"""
        # Проверка на допустимую частоту дискретизации
        if self.sample_rate not in [8000, 16000, 32000, 48000]:
            raise ValueError("Sample rate must be 8000, 16000, 32000 or 48000")
        
        # Если нужно, преобразуем аудио к нужной частоте дискретизации
        if self.sample_rate != 16000:
            audio_resampled = self._resample_audio(self.audio, self.sample_rate, 16000)
            vad_sample_rate = 16000
        else:
            audio_resampled = self.audio
            vad_sample_rate = self.sample_rate
            
        # Нормализация аудио к диапазону [-1, 1]
        if audio_resampled.dtype != np.int16:
            audio_int16 = self._float_to_int16(audio_resampled)
        else:
            audio_int16 = audio_resampled
            
        # Размер окна в миллисекундах (10, 20 или 30 мс)
        window_duration_ms = 10
        window_size = int(vad_sample_rate * window_duration_ms / 1000)
        
        # Разделение аудио на окна
        windows = []
        for i in range(0, len(audio_int16) - window_size, window_size):
            window = audio_int16[i:i + window_size]
            windows.append(window)
            
        # Детекция речи для каждого окна
        speech_windows = []
        for i, window in enumerate(windows):
            if len(window) == window_size:
                try:
                    is_speech = self.vad.is_speech(window.tobytes(), vad_sample_rate)
                    speech_windows.append(is_speech)
                except:
                    speech_windows.append(False)
            else:
                speech_windows.append(False)
                
        return speech_windows
    
    def convert_windows_to_readable_labels(self, speech_windows):
        """Конвертация бинарных меток в читаемые сегменты"""
        window_duration_ms = 10
        window_size_samples = int(16000 * window_duration_ms / 1000)
        
        # Конвертация частоты дискретизации обратно к оригинальной
        sample_rate_ratio = self.sample_rate / 16000
        window_size_original_sr = int(window_size_samples * sample_rate_ratio)
        
        segments = []
        start_speech = None
        
        for i, is_speech in enumerate(speech_windows):
            if is_speech and start_speech is None:
                start_speech = i
            elif not is_speech and start_speech is not None:
                # Конец сегмента речи
                start_sample = int(start_speech * window_size_original_sr)
                end_sample = int((i + 1) * window_size_original_sr)
                
                segments.append({
                    'speech_begin': start_sample / self.sample_rate,
                    'speech_end': end_sample / self.sample_rate,
                    'speech_begin_ids': start_sample,
                    'speech_end_ids': end_sample
                })
                start_speech = None
                
        # Обработка случая, когда речь заканчивается в конце аудио
        if start_speech is not None:
            start_sample = int(start_speech * window_size_original_sr)
            end_sample = len(self.audio)
            
            segments.append({
                'speech_begin': start_sample / self.sample_rate,
                'speech_end': end_sample / self.sample_rate,
                'speech_begin_ids': start_sample,
                'speech_end_ids': end_sample
            })
            
        return segments
    
    def _resample_audio(self, audio: np.ndarray, original_sr: int, target_sr: int) -> np.ndarray:
        """Переоценка частоты дискретизации аудио"""
        if original_sr == target_sr:
            return audio
            
        duration = len(audio) / original_sr
        num_samples = int(duration * target_sr)
        
        # Используем scipy для ресемплинга
        resampled = signal.resample(audio, num_samples)
        return resampled
    
    def _float_to_int16(self, audio: np.ndarray) -> np.ndarray:
        """Конвертация float аудио в int16"""
        # Нормализация к диапазону [-1, 1] если нужно
        if np.max(np.abs(audio)) > 1.0:
            audio = audio / np.max(np.abs(audio))
            
        audio_int16 = (audio * 32767).astype(np.int16)
        return audio_int16