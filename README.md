# Proslushka

ASR модель для работы с голосовыми сообщениями. Реализованы модели:
- [QuartzNet15x5](https://arxiv.org/pdf/1910.10261)
- [CitriNet](https://arxiv.org/pdf/2104.01721)

## Data
Для обучения модели необходимо подготовить датафрейм вида:

| audio_filepath            | text                               | duration  |
|:--------------------------|:-----------------------------------|:----------|
| train_opus/crowd/0/a.opus | пример сообщения на русском языке  | 4.25      |

- `audio_filepath` - путь до аудиозаписи
- `text` - нормализаванный текст (допустимые значения из [CHARS](src/features/tokenizer.py))
- `duration` - время аудиозаписи в секундах (необходимо для семплирования в [BatchSampler](src/features/dataset.py))

## Model
За основу QuartzNet взята реализация модели от [Сбера](https://github.com/salute-developers/golos/tree/master)
на датасете `Golos`. Гиперпараметры препроцессинга перенесены все за исключением нормализации.
[Код](src/utils/transfer_learning.py) для переноса весов.

Дополнительно проводились эксперименты с добавлением `SqueezeExcite` в конец блоков `Jasper`.
Идея возникла из схожей с QuartzNet архитектуры - [Citrinet](https://arxiv.org/pdf/2104.01721).
По итогу нет однозначного улучшения, часть метрик на валидации лучше, но появляются "лишние"
звуки (как пример: "годные" перешло в "глодные"). Как вариант можно расширить словарь токенов.

Модель CitriNet показала результаты лучше как по WER, так и по времени обработки аудио. Единственная проблема в том,
что на данный момент нет обученных весов для русского языка (за исключением citrinet_1024 с более 140M параметров),
поэтому нужно искать [веса для английского языка](https://huggingface.co/nvidia/stt_en_citrinet_384_ls).

## Inference
Обученную модель в формате `ONNX` предполагается использовать на CPU в режиме `serverless`
(aka `AWS Lambda`). Из-за ограничений платформы для деплоя часть функций из `librosa` перенесена
в отдельный [файл](src/app/librosa.py).

Для приведения разных форматов аудио (sample_rate, codec, mono, bitrate, ...) к данным 
из обучения используется бинарник [ffmpeg](https://ffmpeg.org/download.html) (Linux Static Builds).

## Commands

```shell
# конвертировать wav в opus формат
sudo apt-get install parallel
sudo apt install ffmpeg
find . -type f -name "*.wav" | parallel ffmpeg -i {} -c:a libvorbis -q:a 2 {.}.ogg

# перенос весов от модели Сбера (без включения SqueezeExcite)
python -m src.utils.transfer_learning \
    --fitted_nemo_model=models/QuartzNet15x5_golos.nemo \
    --save_path=models/quartznet15x5_sber_transfer.state_dict

# обучение модели
python -m src.model.train_model \
    --model=[quartznet|citrinet] \
    --dataset_path=/.../golos_opus/train_opus \
    --train_manifest=/.../golos_opus/train_opus/1hour.jsonl \
    --valid_manifest=/.../golos_opus/train_opus/1hour.jsonl \
    --batch_size=16 \
    --max_epochs=1 \
    --accumulate_grad_batches=256
```