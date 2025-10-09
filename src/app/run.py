import os
import json
import logging
import subprocess
import sys
import onnxruntime
import soundfile as sf

from aiogram import Bot, types
from aiogram import Dispatcher

from tokenizers import Tokenizer

import src.app.lib.asr as asr

citrinet_model = onnxruntime.InferenceSession("models/citrinet_model.onnx")
citrinet_tokenizer = Tokenizer.from_file("models/citrinet_tokenizer.json")

try:
    comma_model = onnxruntime.InferenceSession("models/comma_model.onnx")
except:
    comma_model = None
distilrubert_tokenizer = Tokenizer.from_file("models/distilrubert_tokenizer.json")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


async def welcome_start(message):
    await message.answer('Hello!\nSend audio')


async def read_voice(message: types.Message):
    try:
        if message.voice.duration > 1800:
            await message.reply(
                'voice duration limit: 30 minutes'
            )
            return
        
        file_path = f'/tmp/{message.voice.file_id}'
        ogg_file_path = f'/tmp/{message.voice.file_id}a.ogg'
        voice_extention = None
        
        await message.voice.download(destination_file=file_path)
        
        for ext in ('ogg', 'm4a'):
            ext_file_path = f'/tmp/{message.voice.file_id}.{ext}'
            os.rename(file_path, ext_file_path)
            file_path = ext_file_path

            try:
                ffmpeg_result = subprocess.run(
                    ['ffmpeg', '-i', file_path, '-c:a', 'libvorbis', '-q:a', '6', '-v', '16', ogg_file_path],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
                logging.info(f"stdout: {ffmpeg_result.stdout}\nstderr: {ffmpeg_result.stderr}")
                audio, sample_rate = sf.read(ogg_file_path)
                voice_extention = ext
                break
            except:
                pass

        if voice_extention is None:
            await message.reply(
                'voice format not in supported: .ogg, .m4a'
            )
            return
    
        asr_model = asr.ASRModel(citrinet_model, citrinet_tokenizer, comma_model, distilrubert_tokenizer)
        decoded_speech = asr_model.speech_to_text(audio, sample_rate)

        if len(decoded_speech) < 1_000:
            await message.reply(
                f'message duration: {message.voice.duration},\n{decoded_speech}'
            )
        else:
            with open(f'/tmp/{message.voice.file_id}.txt', mode='w', encoding='utf-8') as file:
                file.write(decoded_speech)
            await message.answer_document(document=types.InputFile(f'/tmp/{message.voice.file_id}.txt'))
    except Exception as e:
        await message.reply(str(e))


async def read_audio(message: types.Message):
    try:
        if message.audio.duration > 1800:
            await message.reply(
                'audio duration limit: 30 minutes'
            )
            return
        
        _, file_extension = os.path.splitext(message.audio.file_name)
        file_path = f'/tmp/{message.audio.file_name}'
        
        await message.audio.download(destination_file=file_path)

        if file_extension not in ('.ogg', '.mp3'):
            ogg_file_path = f'/tmp/{message.audio.file_id}.ogg'
            # https://yandex.cloud/ru/docs/functions/tutorials/video-converting-queue
            subprocess.run(['ffmpeg', '-i', file_path, '-c:a', 'libvorbis', '-q:a', '6', '-v', '16', ogg_file_path])
            logging.info('audio converted')
            file_path = ogg_file_path

        asr_model = asr.ASRModel(citrinet_model, citrinet_tokenizer, comma_model, distilrubert_tokenizer)
        decoded_speech = asr_model.speech_to_text(*sf.read(file_path))

        if len(decoded_speech) < 1_000:
            await message.reply(
                f'message duration: {message.audio.duration},\n{decoded_speech}'
            )
        else:
            with open(f'/tmp/{message.audio.file_id}.txt', mode='w', encoding='utf-8') as file:
                file.write(decoded_speech)
            await message.answer_document(document=types.InputFile(f'/tmp/{message.audio.file_id}.txt'))
    except Exception as e:
        message.reply(str(e))


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
