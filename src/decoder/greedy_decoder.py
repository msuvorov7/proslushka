import torch


class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, tokenizer, blank):
        super().__init__()
        self.tokenizer = tokenizer
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> list:
        """Given a sequence emission over labels, get the best path
        Args:
          emission (Tensor): Logit tensors. Shape `[batch_size, num_seq, num_label]`.

        Returns:
          List[str]: The resulting transcript
        """
        indices = torch.argmax(emission, dim=-1).detach().cpu().numpy()
        decoded = self.tokenizer.decode_batch(indices)
        return [item.replace(' ##', '') for item in decoded]
        
    def decode_text(self, real_text: torch.Tensor) -> list:
        indices = real_text.detach().cpu().numpy()
        decoded = self.tokenizer.decode_batch(indices)
        return [item.replace(' ##', '') for item in decoded]
