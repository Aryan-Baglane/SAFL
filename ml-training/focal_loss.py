import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 1.0, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce = F.cross_entropy(logits, targets, reduction="none")
        pt = torch.exp(-ce)
        loss = self.alpha * (1 - pt) ** self.gamma * ce
        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss

def batch_hard_triplet_loss(embeddings: torch.Tensor, labels: torch.Tensor, margin: float = 0.3) -> torch.Tensor:
    """
    Hard positive / hard negative mining within a batch.
    embeddings: [B, D]
    labels: [B]
    """
    if embeddings.size(0) < 3:
        return embeddings.new_tensor(0.0)

    embeddings = F.normalize(embeddings, p=2, dim=-1)
    dist = torch.cdist(embeddings, embeddings, p=2)

    labels = labels.unsqueeze(1)
    pos_mask = labels.eq(labels.T)
    neg_mask = ~pos_mask

    losses = []
    for i in range(dist.size(0)):
        pos = dist[i][pos_mask[i]]
        neg = dist[i][neg_mask[i]]
        pos = pos[pos > 0]  # exclude self
        if pos.numel() == 0 or neg.numel() == 0:
            continue
        hardest_pos = pos.max()
        hardest_neg = neg.min()
        losses.append(F.relu(hardest_pos - hardest_neg + margin))

    if not losses:
        return embeddings.new_tensor(0.0)
    return torch.stack(losses).mean()