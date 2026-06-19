import random
import numpy as np

class SmoothLLMDetector:
    def __init__(self, perturbation_pct=0.10, num_perturbations=10):
        self.perturbation_pct = perturbation_pct
        self.num_perturbations = num_perturbations
        self.chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    def _perturb_string(self, text: str) -> str:
        if not text:
            return text
        text_list = list(text)
        num_to_swap = max(1, int(len(text_list) * self.perturbation_pct))
        sampled_indices = random.sample(range(len(text_list)), min(num_to_swap, len(text_list)))
        for idx in sampled_indices:
            if text_list[idx].isalnum():
                text_list[idx] = random.choice(self.chars)
        return "".join(text_list)

    def evaluate_ensemble(self, prompt: str, classification_fn) -> dict:
        """
        Generates N perturbations, executes the local MiniLM classifier against each,
        and extracts the mean probability, structural variance, and absolute attack vote ratio.
        """
        variations = [self._perturb_string(prompt) for _ in range(self.num_perturbations)]
        probabilities = []
        attack_votes = 0
        
        for var in variations:
            # classification_fn must return (highest_attack_probability, is_attack_boolean)
            prob, is_attack = classification_fn(var)
            probabilities.append(prob)
            if is_attack:
                attack_votes += 1
                
        prob_array = np.array(probabilities)
        mean_prob = float(np.mean(prob_array))
        variance_val = float(np.var(prob_array))
        vote_ratio = float(attack_votes / self.num_perturbations)
        
        return {
            "mean_probability": mean_prob,
            "variance": variance_val,
            "attack_vote_ratio": vote_ratio
        }
