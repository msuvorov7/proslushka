from torch.optim.lr_scheduler import _LRScheduler
from math import cos, pi


class WarmupCosLR(_LRScheduler):
    def __init__(
        self,
        optimizer,
        max_iter,
        warmup_factor=1.0 / 3,
        warmup_iters=500,
        last_epoch=-1,
    ):
        self.max_iter = max_iter
        self.warmup_factor = warmup_factor
        self.warmup_iters = warmup_iters
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        warmup_factor = 1
        if self.last_epoch < self.warmup_iters:
            alpha = float(self.last_epoch) / self.warmup_iters
            warmup_factor = self.warmup_factor * (1 - alpha) + alpha

        lr_coeff =  (1 + cos(pi * self.last_epoch / self.max_iter)) / 2
        
        return [
            base_lr
            * warmup_factor
            * lr_coeff
            for base_lr in self.base_lrs
        ]
