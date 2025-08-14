import os
import torch
from tokenizers import Tokenizer

import src.features.tokenizer as tokenizer

import src.model.torch.quartznet as quartznet
import src.model.torch.citrinet as citrinet
import src.model.torch.conformer as conformer


class ASRModel:
    def __init__(
            self,
            model_name: str,
            corpus: list[str],
            pre_trained_sd: str = None,
            pre_trained_tokenizer: str = None,
        ):

        tokenizer_config = {
            'quartznet': tokenizer.QUARTZNET_TOKENIZER,
            'citrinet': tokenizer.CITRINET_TOKENIZER,
            'conformer': tokenizer.CONFORMER_TOKENIZER,
        }

        self.model_name = model_name

        if pre_trained_tokenizer:
            self.model_tokenizer = Tokenizer.from_file(pre_trained_tokenizer)
        else:
            self.model_tokenizer = tokenizer_config[model_name]
        self.train_tokenizer(corpus)
        
        asr_config = {
            'quartznet': {
                'input_scale': 2,
                'model': quartznet.QuartzNet(
                    R_repeat=2,
                    in_channels=64,
                    include_se_block=False,
                    out_channels=self.model_tokenizer.get_vocab_size() + 1,  # plus one for blank token (ctc loss)
                ),
                'n_mels': 64,
            },
            'citrinet': {
                'input_scale': 8,
                'model': citrinet.CitriNet(
                    K=4,
                    C=384,
                    R_repeat=4,
                    Gamma=8,
                    in_channels=80,
                    out_channels=self.model_tokenizer.get_vocab_size() + 1,  # plus one for blank token (ctc loss)
                ),
                'n_mels': 80,
            },
            'conformer': {
                'input_scale': 4,
                'model': conformer.Conformer(
                    in_dim=1,
                    n_mels=80,
                    encoder_dim=176,
                    num_blocks=16,
                    num_heads=4,
                    dropout=0.1,
                    out_dim=self.model_tokenizer.get_vocab_size() + 1,  # plus one for blank token (ctc loss)
                ),
                'n_mels': 80,
            },
        }

        self.asr_model = asr_config[model_name]

        if pre_trained_sd:
            self.asr_model['model'].load_state_dict(torch.load(pre_trained_sd))

    def train_tokenizer(self, corpus: list[str]):
        os.environ['TOKENIZERS_PARALLELISM'] = 'true'
        if self.model_name == 'quartznet':
            pass
        elif self.model_name == 'citrinet':
            self.model_tokenizer.train_from_iterator(
                corpus,
                tokenizer.CITRINET_TRAINER,
            )
        elif self.model_name == 'conformer':
            self.model_tokenizer.train_from_iterator(
                corpus,
                tokenizer.CONFORMER_TRAINER,
            )
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'