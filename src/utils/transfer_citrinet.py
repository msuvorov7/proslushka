import logging
import sys

import torch
import argparse
import nemo.collections.asr as nemo_asr

from model.torch.citrinet import CitriNet


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)


def init_weights(asr_model, model):
    C1_params = dict()
    for model_param, nemo_param in zip(model.C1.state_dict(), asr_model.encoder.encoder[0].state_dict()):
        print(model_param, ' <- ', nemo_param)
        if 'fc.0.weight' in model_param or 'fc.2.weight' in model_param:
            C1_params[model_param] = asr_model.encoder.encoder[0].state_dict()[nemo_param].unsqueeze(-1)
        else:
            C1_params[model_param] = asr_model.encoder.encoder[0].state_dict()[nemo_param]
    model.C1.load_state_dict(C1_params)

    C2_params = dict()
    for model_param, nemo_param in zip(model.C2.state_dict(), asr_model.encoder.encoder[-1].state_dict()):
        print(model_param, nemo_param)
        if 'fc.0.weight' in model_param or 'fc.2.weight' in model_param:
            C2_params[model_param] = asr_model.encoder.encoder[-1].state_dict()[nemo_param].unsqueeze(-1)
        else:
            C2_params[model_param] = asr_model.encoder.encoder[-1].state_dict()[nemo_param]
    model.C2.load_state_dict(C2_params)

    B1_params = dict()
    for model_param, nemo_param in zip(model.megablock1.state_dict(), asr_model.encoder.encoder[1:7].state_dict()):
        print(model_param, nemo_param)
        if 'fc.0.weight' in model_param or 'fc.2.weight' in model_param:
            B1_params[model_param] = asr_model.encoder.encoder[1:7].state_dict()[nemo_param].unsqueeze(-1)
        else:
            B1_params[model_param] = asr_model.encoder.encoder[1:7].state_dict()[nemo_param]
    model.megablock1.load_state_dict(B1_params)

    B2_params = dict()
    for model_param, nemo_param in zip(model.megablock2.state_dict(), asr_model.encoder.encoder[7:14].state_dict()):
        print(model_param, nemo_param)
        if 'fc.0.weight' in model_param or 'fc.2.weight' in model_param:
            B2_params[model_param] = asr_model.encoder.encoder[7:14].state_dict()[nemo_param].unsqueeze(-1)
        else:
            B2_params[model_param] = asr_model.encoder.encoder[7:14].state_dict()[nemo_param]
    model.megablock2.load_state_dict(B2_params)

    B3_params = dict()
    for model_param, nemo_param in zip(model.megablock3.state_dict(), asr_model.encoder.encoder[14:22].state_dict()):
        print(model_param, nemo_param)
        if 'fc.0.weight' in model_param or 'fc.2.weight' in model_param:
            B3_params[model_param] = asr_model.encoder.encoder[14:22].state_dict()[nemo_param].unsqueeze(-1)
        else:
            B3_params[model_param] = asr_model.encoder.encoder[14:22].state_dict()[nemo_param]
    model.megablock3.load_state_dict(B3_params)

    return model.state_dict()


if __name__ == '__main__':
    args_parser = argparse.ArgumentParser()
    args_parser.add_argument('--fitted_nemo_model', default='nvidia/stt_en_citrinet_384_ls', dest='fitted_nemo_model')
    args_parser.add_argument('--save_path', default='models/citrinet_384_transfer.state_dict', dest='save_path')
    args = args_parser.parse_args()

    fitted_citrinet = nemo_asr.models.EncDecCTCModelBPE.from_pretrained(args.fitted_nemo_model)
    dummy_model = CitriNet(
        K=4,
        C=384,
        R_repeat=4,
        Gamma=8,
        in_channels=80,
        out_channels=257,
    )

    fitted_state_dict = init_weights(source_model=fitted_citrinet, target_model=dummy_model)
    torch.save(fitted_state_dict, args.save_path)
    logging.info(f'state_dict saved in {args.save_path}')
