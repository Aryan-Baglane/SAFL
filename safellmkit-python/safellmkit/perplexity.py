import math
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class GPT2PerplexityScorer:
    def __init__(self, model_name: str = "gpt2"):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)
        self.model.eval()
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    @torch.no_grad()
    def perplexity(self, text: str) -> float:
        if not text or not text.strip():
            return 1.0

        enc = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=256,
            padding=False
        )
        out = self.model(**enc, labels=enc["input_ids"])
        loss = float(out.loss.item())
        return float(math.exp(loss))

    def normalized_score(self, text: str) -> float:
        """
        Map perplexity into 0..1. Higher means more anomalous.
        """
        pp = self.perplexity(text)
        log_pp = math.log(max(pp, 1e-6))
        low = math.log(10.0)
        high = math.log(500.0)
        return max(0.0, min(1.0, (log_pp - low) / (high - low)))