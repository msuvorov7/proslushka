import torch.nn as nn


class CTCLoss(nn.Module):
    """
    Usual CTCLoss from torch, but with retain_graph=True for pytorch-lightning backward.
    """
    def __init__(self, blank):
        super().__init__()
        self.loss = None
        self.criterion = nn.CTCLoss(blank=blank, zero_infinity=True)

    def forward(self, output, texts, input_len, target_len):
        self.loss = self.criterion(output, texts, input_len, target_len)
        return self.loss

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss
