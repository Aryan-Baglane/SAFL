import random
from dataclasses import dataclass
from typing import Callable, List

@dataclass
class SmoothResult:
    mean_score: float
    variance: float
    attack_vote_ratio: float
    scores: List[float]

class SmoothLLMEnsemble:
    def __init__(self, predictor: Callable[[str], float], trials: int = 12, seed: int = 7):
        self.predictor = predictor
        self.trials = trials
        self.rng = random.Random(seed)

    def _perturb(self, text: str) -> str:
        if not text:
            return text

        chars = list(text)
        choice = self.rng.randint(0, 3)

        if choice == 0 and len(chars) >= 2:
            i = self.rng.randint(0, len(chars) - 2)
            chars[i], chars[i + 1] = chars[i + 1], chars[i]
        elif choice == 1 and len(chars) > 3:
            i = self.rng.randint(0, len(chars) - 1)
            del chars[i]
        elif choice == 2:
            i = self.rng.randint(0, len(chars))
            chars.insert(i, self.rng.choice([" ", ".", ",", "e", "a"]))
        else:
            i = self.rng.randint(0, len(chars) - 1)
            chars[i] = self.rng.choice([chars[i].lower(), chars[i].upper(), "@", "3", "1"])

        return "".join(chars)

    def score(self, text: str, threshold: float = 0.5) -> SmoothResult:
        scores = [float(self.predictor(text))]
        for _ in range(self.trials - 1):
            perturbed = self._perturb(text)
            scores.append(float(self.predictor(perturbed)))

        mean_score = sum(scores) / len(scores)
        variance = sum((s - mean_score) ** 2 for s in scores) / len(scores)
        attack_votes = sum(1 for s in scores if s >= threshold)
        return SmoothResult(
            mean_score=mean_score,
            variance=variance,
            attack_vote_ratio=attack_votes / len(scores),
            scores=scores
        )