import torch

import numpy as np


class GreedyCTCDecoder:
    def __init__(self, labels: dict, blank: int):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def decode_transcription(self, emission: torch.Tensor) -> list:
        """
        :param emission: Logit tensors. Shape `[batch_size, num_seq, num_label]`.
        :return: List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1)  # [batch_size, num_seq]
        indices = np.vectorize(
            lambda item: self.labels.get(item, '')
        )(
            indices.detach().cpu().numpy()
        )
        joined = np.apply_along_axis(lambda item: ''.join(item).strip(), -1, indices)

        return joined.tolist()

    def decode_text(self, real_text: torch.Tensor) -> list:
        indices = np.vectorize(
            lambda item: self.labels.get(item, '')
        )(
            real_text.detach().cpu().numpy()
        )
        joined = np.apply_along_axis(lambda item: ''.join(item).strip(), -1, indices)
        return joined.tolist()
