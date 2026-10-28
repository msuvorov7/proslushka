import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import MetricCollection
from torchmetrics.text import WordErrorRate, CharErrorRate

from pytorch_optimizer.optimizer.novograd import NovoGrad

from src.model.scheduler import WarmupCosLR


class ASRLightning(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            decoder: nn.Module,
            t_max: int,
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
        
        self.save_hyperparameters(ignore=['model', 'criterion', 'decoder'])
        
        metric = MetricCollection({'wer': WordErrorRate(), 'cer': CharErrorRate()})
        self.train_metric = metric.clone(prefix='train/')
        self.valid_metric = metric.clone(prefix='valid/')

    def forward(self, x, lengths):
        return self.model(x, lengths)
    
    def evaluate(self, batch, mode: str) -> tuple:
        melspecs, texts, input_lengths, label_lengths = batch

        # Forward pass
        # Выход модели: [batch, channels, time] -> [time, batch, channels] для CTCLoss
        output = self.model(melspecs, input_lengths).permute(2, 0, 1)
        output = F.log_softmax(output, dim=2)
        
        # Безопасный расчет длин для CTCLoss
        out_lengths = (input_lengths.float() / self.inputs_length_scale).ceil().int()
        # Защита: длина не может быть больше реального размера тензора
        out_lengths = torch.clamp(out_lengths, max=output.shape[0])
        
        # Считаем лосс
        loss = self.criterion(
            output,
            texts,
            out_lengths,
            label_lengths,
        )
        
        # Считаем метрики (только для валидации, чтобы не тормозить трейн)
        if mode == 'valid':
            metric_func = self.valid_metric
            
            # Возвращаем к формату [batch, time, channels] для декодера
            output_batch_first = output.permute(1, 0, 2)
            
            predicted_texts = self.decoder(output_batch_first)
            real_texts = self.decoder.decode_text(texts)
            
            valid_preds = []
            valid_reals = []
            
            for pred, true in zip(predicted_texts, real_texts):
                # Игнорируем только битую разметку. 
                # Если true != "", а pred == "", мы ОБЯЗАНЫ добавить это в метрику
                if true != "": 
                    valid_preds.append(pred)
                    valid_reals.append(true)
            
            if len(valid_preds) > 0:
                metric = metric_func(valid_preds, valid_reals)
            else:
                # Если весь батч был битый (что странно)
                metric = metric_func(['error'], ['error'])
                
            return loss, metric
        
        return loss, None
    
    def training_step(self, batch, batch_idx):
        train_loss, _ = self.evaluate(batch, 'train')
        self.log("train_loss", train_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        return train_loss

    def validation_step(self, batch, batch_idx, dataloader_idx=0):
        valid_loss, valid_metric = self.evaluate(batch, 'valid')
        self.log("valid_loss", valid_loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)
        self.log_dict(valid_metric, on_step=False, on_epoch=True, prog_bar=True, logger=True)
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
            warmup_iters=min(1000, int(self.t_max * 0.05)), # Греем первые 5% шагов, но не больше 1000
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
