import os
import json
import logging

from aiogram import Bot, types
from aiogram import Dispatcher

import soundfile as sf
import numpy as np
import onnxruntime

from librosa import melspectrogram, resample, preemphasis

onnx_model = onnxruntime.InferenceSession('models/quartznet_15x5.onnx')
log = logging.getLogger(__name__)


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


async def speech_to_text(message: types.Message):
    file_path = f'/tmp/{message.voice.file_id}.ogg'
    await message.voice.download(destination_file=file_path)

    wav, sr = sf.read(file_path)
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
    dp.register_message_handler(speech_to_text, content_types=[types.ContentType.VOICE])

    log.debug('Handlers are registered.')


async def process_event(event, dp: Dispatcher):
    """
    Converting an Yandex.Cloud functions event to an update and
    handling tha update.
    """

    update = json.loads(event['body'])
    log.debug('Update: ' + str(update))

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
