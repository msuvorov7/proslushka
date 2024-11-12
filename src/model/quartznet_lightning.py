import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import MetricCollection
from torchmetrics.text import WordErrorRate, CharErrorRate


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
    def __init__(self, model, criterion, decoder):
        super().__init__()

        self.save_hyperparameters()
        self.model = model
        self.criterion = criterion
        self.decoder = decoder

        self.save_hyperparameters(ignore=['model', 'criterion'])

        metric = MetricCollection({'wer': WordErrorRate(), ' cer': CharErrorRate()})
        self.train_metric = metric.clone(prefix='train/')
        self.valid_metric = metric.clone(prefix='valid/')

    def forward(self, x):
        return self.model(x)

    def evaluate(self, batch, metric_func):
        melspecs = batch[0]
        texts = batch[1]
        input_lengths = batch[2]
        label_lengths = batch[3]

        output = self.model(melspecs).permute(2, 0, 1)
        output = F.log_softmax(output, dim=2)

        # eval loss
        loss = self.criterion(
            output,
            texts,
            torch.ceil(input_lengths.float() / 2).int(),
            label_lengths,
        )

        # eval metrics
        output = output.permute(1, 0, 2)
        predicted_text = []
        real_text = []
        for pred, true in zip(self.decoder.decode_transcription(output), self.decoder.decode_text(texts)):
            if (pred != "") and (true != ""):
                predicted_text.append(pred)
                real_text.append(true)

        if (len(predicted_text) > 0) and (len(real_text) > 0):
            metric = metric_func(predicted_text, real_text)
        else:
            metric = metric_func(['error'], ['error'])

        return loss, metric

    def training_step(self, batch, batch_idx):
        train_loss, train_metric = self.evaluate(batch, self.train_metric)
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(train_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx):
        valid_loss, valid_metric = self.evaluate(batch, self.valid_metric)
        self.log("valid_loss", valid_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(valid_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return valid_loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=5e-4, weight_decay=0.0001)
        return optimizer
