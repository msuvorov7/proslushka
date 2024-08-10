import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl


class CustomCTCLoss(nn.Module):
    """
    Usual CTCLoss from torch, but with retain_graph=True for pytorch-lightning backward.
    """
    def __init__(self, blank):
        super(CustomCTCLoss, self).__init__()
        self.loss = None
        self.criterion = nn.CTCLoss(blank=blank, zero_infinity=True)

    def forward(self, output, texts, input_len, target_len):
        self.loss = self.criterion(output, texts, input_len, target_len)
        return self.loss

    def backward(self, retain_graph=True):
        self.loss.backward(retain_graph=retain_graph)
        return self.loss


class QuartzNetLightning(pl.LightningModule):
    def __init__(self, model, criterion):
        super().__init__()

        self.save_hyperparameters()
        self.model = model
        self.criterion = criterion

    def forward(self, x):
        return self.model(x)

    def evaluate(self, batch):
        melspecs = batch[0]
        texts = batch[1]
        input_lengths = batch[2]
        label_lengths = batch[3]

        output = self.model(melspecs).permute(2, 0, 1)
        output = F.log_softmax(output, dim=2)
        loss = self.criterion(
            output,
            texts,
            torch.ceil(input_lengths.float() / 2).int(),
            label_lengths,
        )

        return loss

    def training_step(self, batch, batch_idx):
        train_loss = self.evaluate(batch)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        valid_loss = self.evaluate(batch)
        self.log("valid_loss", valid_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return valid_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=0.0001)
        return optimizer
