import os
import json
import logging
import subprocess
import sys

from aiogram import Bot, types
from aiogram import Dispatcher

import soundfile as sf
import numpy as np
import onnxruntime

from librosa import melspectrogram, resample, preemphasis

onnx_model = onnxruntime.InferenceSession('models/quartznet_15x5.onnx')
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


chars = [
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
int_to_char = dict(enumerate(chars))
char_to_int = {v: k for k, v in int_to_char.items()}


async def welcome_start(message):
    await message.answer('Hello!\nSend audio')


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

    out = tmp - out
    return out


async def read_voice(message: types.Message):
    file_path = f'/tmp/{message.voice.file_id}.ogg'
    # opus_file_path = f'/tmp/{message.voice.file_id}.opus'
    await message.voice.download(destination_file=file_path)

    # subprocess.run(['ffmpeg', '-i', file_path, '-c:a', 'libopus', '-b:a', '36k', '-ac', '1', '-v', '16', opus_file_path])
    logging.info('audio converted')

    decoded = speech_to_text(file_path)

    if len(decoded) < 1_000:
        await message.reply(
            f'message duration: {message.voice.duration},\n{decoded}'
        )
    else:
        with open(f'/tmp/{message.voice.file_id}.txt', mode='w', encoding='utf-8') as file:
            file.write(decoded)
        await message.answer_document(document=types.InputFile(f'/tmp/{message.voice.file_id}.txt'))


async def read_audio(message: types.Message):
    file_path = f'/tmp/{message.audio.file_name}'
    # opus_file_path = f'/tmp/{message.audio.file_id}.opus'
    await message.audio.download(destination_file=file_path)

    # https://yandex.cloud/ru/docs/functions/tutorials/video-converting-queue
    # subprocess.run(['ffmpeg', '-i', file_path, '-c:a', 'libopus', '-b:a', '36k', '-ac', '1', '-v', '16', opus_file_path])
    logging.info('audio converted')

    decoded = speech_to_text(file_path)

    if len(decoded) < 1_000:
        await message.reply(
            f'message duration: {message.audio.duration},\n{decoded}'
        )
    else:
        with open(f'/tmp/{message.audio.file_id}.txt', mode='w', encoding='utf-8') as file:
            file.write(decoded)
        await message.answer_document(document=types.InputFile(f'/tmp/{message.audio.file_id}.txt'))


def speech_to_text(file_path: str) -> str:
    wav, sr = sf.read(file_path)
    # stereo to mono
    if wav.ndim == 2:
        wav = wav.mean(axis=1)
    wav = resample(wav, orig_sr=sr, target_sr=16000)
    wav = preemphasis(wav, coef=0.97)

    mel_spec = melspectrogram(
        y=wav,
        sr=16_000,
        n_fft=512,
        hop_length=160,
        win_length=320,
        window='hann',
        pad_mode='reflect',
        power=2,
        fmin=0,
        n_mels=64,
        norm='slaney',
        center=True
    )

    model_input = {onnx_model.get_inputs()[0].name: np.log(mel_spec)[np.newaxis, :, :].astype(np.float32)}
    model_output = onnx_model.run(None, model_input)[0]
    output = model_output.transpose(2, 0, 1)
    output = log_softmax(output, axis=2)
    output = output.transpose(1, 0, 2)

    decoded = ''.join([int_to_char.get(i, '') for i in output.argmax(axis=2).tolist()[0]])

    await message.reply(
        f'message duration: {message.voice.duration},\n{sr},\n{mel_spec.shape}\n{decoded}'
    )


# Functions for Yandex.Cloud
async def register_handlers(dp: Dispatcher):
    """Registration all handlers before processing update."""

    dp.register_message_handler(welcome_start, commands=['start'])
    dp.register_message_handler(read_voice, content_types=[types.ContentType.VOICE])
    dp.register_message_handler(read_audio, content_types=[types.ContentType.AUDIO])

    logging.debug('Handlers are registered.')


async def process_event(event, dp: Dispatcher):
    """
    Converting an Yandex.Cloud functions event to an update and
    handling tha update.
    """

    update = json.loads(event['body'])
    logging.debug('Update: ' + str(update))

    Bot.set_current(dp.bot)
    update = types.Update.to_object(update)
    await dp.process_update(update)


async def handler(event, context):
    """Yandex.Cloud functions handler."""

    if event['httpMethod'] == 'POST':
        # Bot and dispatcher initialization
        bot = Bot(os.environ.get('TELEGRAM_BOT_TOKEN'))
        dp = Dispatcher(bot)

        await register_handlers(dp)
        await process_event(event, dp)

        return {'statusCode': 200, 'body': 'ok'}
    return {'statusCode': 405}
