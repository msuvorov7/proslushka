import torch
from itertools import groupby


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, tokenizer, blank: int = 0):
        super().__init__()
        self.tokenizer = tokenizer
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> list:
        """
        Декодирование предсказаний модели.
        Args:
          emission (Tensor): Логиты. Формат `[batch_size, time, num_labels]`.
        Returns:
          List[str]: Распознанный текст.
        """
        # 1. Берем argmax и переводим в обычные списки Python (это быстрее для groupby)
        indices = torch.argmax(emission, dim=-1).detach().cpu().tolist()
        
        collapsed_indices = []
        for seq in indices:
            # 2. Логика CTC: схлопываем подряд идущие дубликаты
            seq_collapsed = [key for key, _ in groupby(seq)]
            
            # 3. Удаляем BLANK-токены
            seq_cleaned = [idx for idx in seq_collapsed if idx != self.blank]
            
            collapsed_indices.append(seq_cleaned)
            
        # 4. Токенизатор делает всю грязную работу (превращает индексы в текст с пробелами)
        # skip_special_tokens=True гарантирует, что в текст не попадут [UNK] и прочий мусор
        return self.tokenizer.decode_batch(collapsed_indices, skip_special_tokens=True)
        
    def decode_text(self, real_text: torch.Tensor) -> list:
        """
        Декодирование реальной разметки (Ground Truth).
        Здесь CTC-схлопывание НЕ нужно, так как это оригинальный текст.
        """
        indices = real_text.detach().cpu().tolist()
        
        cleaned_indices = []
        for seq in indices:
            # Просто отфильтровываем нули (если батч был добит паддингом)
            seq_cleaned = [idx for idx in seq if idx != self.blank]
            cleaned_indices.append(seq_cleaned)
            
        return self.tokenizer.decode_batch(cleaned_indices, skip_special_tokens=True)
