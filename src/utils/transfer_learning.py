import logging
import sys

import torch
import argparse
import nemo.collections.asr as nemo_asr

from src.model.quartznet_torch import QuartzNet
from src.features.dataset import CHARS


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def init_weights(source_model, target_model):
    C1_params = dict()
    for target_param, source_param in zip(target_model.C1.state_dict(), source_model.encoder.encoder[0].state_dict()):
        C1_params[target_param] = source_model.encoder.encoder[0].state_dict()[source_param]
    target_model.C1.load_state_dict(C1_params)

    C2_params = dict()
    for target_param, source_param in zip(target_model.C2.state_dict(), source_model.encoder.encoder[16].state_dict()):
        C2_params[target_param] = source_model.encoder.encoder[16].state_dict()[source_param]
    target_model.C2.load_state_dict(C2_params)

    C3_params = dict()
    for target_param, source_param in zip(target_model.C3.state_dict(), source_model.encoder.encoder[17].state_dict()):
        C3_params[target_param] = source_model.encoder.encoder[17].state_dict()[source_param]
    target_model.C3.load_state_dict(C3_params)

    C4_params = dict()
    for target_param, source_param in zip(target_model.C4.state_dict(), source_model.decoder.state_dict()):
        C4_params[target_param] = source_model.decoder.state_dict()[source_param]
    target_model.C4.load_state_dict(C4_params)

    B_params = dict()
    for target_param, source_param in zip(target_model.B.state_dict(), source_model.encoder.encoder[1:16].state_dict()):
        B_params[target_param] = source_model.encoder.encoder[1:16].state_dict()[source_param]
    target_model.B.load_state_dict(B_params)

    return target_model.state_dict()


if __name__ == '__main__':
    # https://github.com/salute-developers/golos/tree/master/golos#acoustic-and-language-models
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--fitted_nemo_model', default='models/QuartzNet15x5_golos.nemo', dest='fitted_nemo_model')
    args_parser.add_argument('--save_path', default='models/quartznet15x5_sber_transfer.state_dict', dest='save_path')
    args = args_parser.parse_args()

    fitted_quartznet = nemo_asr.models.EncDecCTCModel.restore_from(args.fitted_nemo_model)
    dummy_model = QuartzNet(
        R_repeat=2,
        in_channels=64,
        out_channels=len(CHARS) + 1,  # plus 1 for blank token (ctc loss)
        include_se_block=False,
    )

    fitted_state_dict = init_weights(source_model=fitted_quartznet, target_model=dummy_model)
    torch.save(fitted_state_dict, args.save_path)
    logging.info(f'state_dict saved in {args.save_path}')
