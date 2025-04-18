import torch
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import MetricCollection
from torchmetrics.text import WordErrorRate, CharErrorRate

from pytorch_optimizer.optimizer.novograd import NovoGrad

from src.model.scheduler import WarmupCosLR


class ASRLightning(pl.LightningModule):
    def __init__(
            self,
            model,
            criterion,
            decoder,
            t_max,
            inputs_length_scale: int,
            lr: float = 0.05,
            weight_decay: float = 0.001,
        ):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.decoder = decoder
        self.t_max = t_max
        self.inputs_length_scale = inputs_length_scale
        self.lr = lr
        self.weight_decay = weight_decay
        
        self.save_hyperparameters(ignore=['model', 'criterion'])
        
        metric = MetricCollection({'wer': WordErrorRate(), ' cer': CharErrorRate()})
        self.train_metric = metric.clone(prefix='train/')
        self.valid_metric = metric.clone(prefix='valid/')

    def forward(self, x):
        return self.model(x)
    
    def evaluate(self, batch, metric_func) -> tuple:
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
            torch.ceil(input_lengths.float() / self.inputs_length_scale).int(),
            label_lengths,
        )
        
        # eval metrics
        output = output.permute(1, 0, 2)
        predicted_text = []
        real_text = []
        for pred, true in zip(self.decoder(output), self.decoder.decode_text(texts)):
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

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        valid_loss, valid_metric = self.evaluate(batch, self.valid_metric)
        self.log("valid_loss", valid_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(valid_metric, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return valid_loss

    def configure_optimizers(self):
        optimizer = NovoGrad(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
            betas=[0.8, 0.25],
        )
        scheduler = WarmupCosLR(
            optimizer=optimizer,
            max_iter=self.t_max,
            warmup_factor=1.0 / 10.0,
            warmup_iters=750,
        )
        
        return (
            [optimizer],
            [
                {
                    'scheduler': scheduler,
                    'interval': 'step',
                    'frequency': 1,
                }
            ]
        )