# Proslushka

ASR model based on [QuartzNet15x5](https://arxiv.org/pdf/1910.10261) for Russian Language.

## Data
Для обучения модели необходимо подготовить датафрейм вида:

| audio_filepath            | text                               | duration  |
|:--------------------------|:-----------------------------------|:----------|
| train_opus/crowd/0/a.opus | пример сообщения на русском языке  | 4.25      |

- `audio_filepath` - путь до аудиозаписи
- `text` - нормализаванный текст (допустимые значения из [CHARS](src/features/dataset.py))
- `duration` - время аудиозаписи в секундах (необходимо для семплирования в [BatchSampler](src/features/dataset.py))

## Model
За основу взята реализация модели от [Сбера](https://github.com/salute-developers/golos/tree/master)
на датасете `Golos`. Гиперпараметры препроцессинга перенесены все за исключением нормализации.
[Код](src/utils/transfer_learning.py) для переноса весов.

Дополнительно проводились эксперименты с добавлением `SqueezeExcite` в конец блоков `Jasper`.
Идея возникла из схожей с QuartzNet архитектуры - [Citrinet](https://arxiv.org/pdf/2104.01721).
По итогу нет однозначного улучшения, часть метрик на валидации лучше, но появляются "лишние"
звуки (как пример: "годные" перешло в "глодные").

## Inference
Обученную модель в формате `ONNX` предполагается использовать на CPU в режиме `serverless`
(aka `AWS Lambda`). Из-за ограничений платформы для деплоя часть функций из `librosa` перенесена
в отдельный [файл](src/app/librosa.py).

Для приведения разных форматов аудио (sample_rate, codec, mono, bitrate, ...) к данным 
из обучения используется бинарник [ffmpeg](https://ffmpeg.org/download.html) (Linux Static Builds).


## Commands

```shell
# перенос весов от модели Сбера (без включения SqueezeExcite)
python -m src.utils.transfer_learning \
    --fitted_nemo_model=models/QuartzNet15x5_golos.nemo \
    --save_path=models/quartznet15x5_sber_transfer.state_dict

# обучение модели
python -m src.model.train_model \
    --train_manifest=data/train_opus/manifest.jsonl \
    --valid_manifest=data/test_opus/crowd/manifest.jsonl \
    --batch_size=32 \
    --max_epochs=15 \
    --accumulate_grad_batches=64 \
    [--checkpoint_path=quartznet_ckpts/last.ckpt]
```