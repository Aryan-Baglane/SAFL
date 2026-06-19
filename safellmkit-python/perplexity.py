import torch
import numpy as np
from transformers import GPT2LMHeadModel, AutoTokenizer

class GPT2PerplexityScorer:
    def __init__(self, model_name="gpt2"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = GPT2LMHeadModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

    def score(self, prompt: str) -> float:
        """
        Computes the true perplexity PP(W) = exp(-1/N * sum(log P(w_i | w_<i))) 
        and normalizes it to a 0..1 scale. High entropy/anomalous text yields scores closer to 1.0.
        """
        if not prompt.strip():
            return 0.0
        
        encodings = self.tokenizer(prompt, return_tensors="pt")
        input_ids = encodings.input_ids.to(self.device)
        target_ids = input_ids.clone()
        
        with torch.no_grad():
            outputs = self.model(input_ids, labels=target_ids)
            neg_log_likelihood = outputs.loss
            
        perplexity = torch.exp(neg_log_likelihood).item()
        
        # Soft-clamping normalization curve to translate raw PP to a 0..1 risk probability metric
        # Base threshold calibrated for normal human prompts vs noisy adversarial suffixes
        normalized_score = 1.0 - (1.0 / (1.0 + np.log1p(max(0.0, perplexity - 15.0)) / 5.0))
        return float(np.clip(normalized_score, 0.0, 1.0))
