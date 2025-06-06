import argparse
import os
import sys
import zipfile


fileDir = os.path.join(os.path.dirname(os.path.realpath(__file__)), '../../')

sys.path.insert(0, os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
))


def create_archive(models_path: str) -> None:
    """
    Создать архив для отправки в S3
    :param models_path: путь до моделей
    :return:
    """
    with zipfile.ZipFile('serverless_functions.zip', mode='w', compression=zipfile.ZIP_DEFLATED) as zf:
        zf.write('src/app/run.py', 'run.py')
        zf.write('src/app/lib/asr.py', 'lib/asr.py')
        zf.write('src/app/lib/librosa.py', 'lib/librosa.py')
        zf.write('src/app/lib/vad.py', 'lib/vad.py')
        zf.write('src/app/requirements.txt', 'requirements.txt')
        zf.write(models_path + 'citrinet_384_10epoch.onnx', 'models/model.onnx')
        zf.write(models_path + 'tokenizer.json', 'models/tokenizer.json')
        zf.write('artifacts/ffmpeg-6.0.1-amd64-static/ffmpeg', 'ffmpeg')


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--models_path', default='models/', dest='models_path')
    args = args_parser.parse_args()

    model_path = fileDir + args.models_path
    create_archive(model_path)
