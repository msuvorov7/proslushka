import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl

from torchmetrics import MetricCollection
from torchmetrics.text import WordErrorRate, CharErrorRate


class ASRLightning(pl.LightningModule):
    def __init__(
            self,
            model: nn.Module,
            criterion: nn.Module,
            decoder: nn.Module,
            t_max: int,
            inputs_length_scale: int,
            optim_config: dict,
        ):
        super().__init__()

        self.model = model
        self.criterion = criterion
        self.decoder = decoder
        self.t_max = t_max
        self.inputs_length_scale = inputs_length_scale
        self.optim_config = optim_config
        
        self.save_hyperparameters(ignore=['model', 'criterion', 'decoder'])
        
        metric = MetricCollection({'wer': WordErrorRate(), 'cer': CharErrorRate()})
        self.train_metric = metric.clone(prefix='train/')
        self.valid_metric = metric.clone(prefix='valid/')

    def forward(self, x, lengths):
        return self.model(x, lengths)
    
    def evaluate(self, batch, mode: str) -> tuple:
        melspecs, texts, input_lengths, label_lengths = batch

        # Forward pass
        # Выход модели: [batch, time, channels] -> [time, batch, channels] для CTCLoss
        logits, out_lengths = self.model(melspecs, input_lengths)
        out_lengths = torch.clamp(out_lengths, max=logits.size(1), min=1)
        # (B, T', V) -> (T', B, V) для CTCLoss
        output = F.log_softmax(logits.transpose(0, 1), dim=2)
        
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
        opt_cfg = self.optim_config['optimizer']
        sch_cfg = self.optim_config['scheduler']

        # Инициализируем оптимизатор
        optimizer = opt_cfg['cls'](self.parameters(), **opt_cfg['kwargs'])

        # Инициализируем шедулер (подменяем T_MAX на реальное значение)
        sched_kwargs = dict(sch_cfg['kwargs'])
        if sched_kwargs.get('max_iter') == 'T_MAX':
            sched_kwargs['max_iter'] = self.t_max
            
        scheduler = sch_cfg['cls'](optimizer=optimizer, **sched_kwargs)

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

    def configure_gradient_clipping(self, optimizer, gradient_clip_val=None, gradient_clip_algorithm=None):
        # Берем клиппинг из рецепта модели. Если там None, клиппинг отключается.
        clip_val = self.optim_config.get('grad_clip', None)
        if clip_val is not None:
            self.clip_gradients(
                optimizer, 
                gradient_clip_val=clip_val, 
                gradient_clip_algorithm=gradient_clip_algorithm
            )

    def on_train_epoch_start(self):
        schedulers = self.lr_schedulers()
        if not schedulers:
            return
        # Lightning возвращает шедулер или список шедулеров
        for sched in schedulers if isinstance(schedulers, list) else [schedulers]:
            if hasattr(sched, "step_epoch"):
                sched.step_epoch(self.current_epoch)
