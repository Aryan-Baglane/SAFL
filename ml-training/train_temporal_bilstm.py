import os
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import Dataset, DataLoader

class TemporalSecurityDataset(Dataset):
    def __init__(self, sequences, labels):
        # sequences shape: [num_samples, seq_len, 384]
        self.sequences = torch.tensor(sequences, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.sequences[idx], self.labels[idx]

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ce = nn.CrossEntropyLoss(reduction='none')

    def forward(self, inputs, targets):
        log_pt = -self.ce(inputs, targets)
        pt = torch.exp(log_pt)
        loss = -self.alpha * ((1.0 - pt) ** self.gamma) * log_pt
        return loss.mean()

class BiLSTMTemporalClassifier(nn.Module):
    def __init__(self, embedding_dim=384, hidden_dim=128, num_classes=3):
        super().__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        # x: [batch, seq_len, 384]
        lstm_out, _ = self.lstm(x) # [batch, seq_len, hidden_dim * 2]
        # Global average pooling over the sequence dimension (dim=1)
        pooled = torch.mean(lstm_out, dim=1) # [batch, hidden_dim * 2]
        return self.classifier(pooled)

def generate_synthetic_temporal_data(num_samples=1000, seq_len=5, embedding_dim=384):
    # Class 0: SAFE (all turns are benign)
    # Class 1: PAIR (turns 0..k-1 are benign, turn k..seq_len-1 are PAIR escalation)
    # Class 2: CRESCENDO (turns 0..k-1 are benign, turn k..seq_len-1 are CRESCENDO escalation)
    
    np.random.seed(42)
    pair_direction = np.random.randn(embedding_dim)
    pair_direction /= np.linalg.norm(pair_direction)
    
    crescendo_direction = np.random.randn(embedding_dim)
    crescendo_direction /= np.linalg.norm(crescendo_direction)
    
    X = np.zeros((num_samples, seq_len, embedding_dim))
    y = np.zeros(num_samples, dtype=int)
    
    for i in range(num_samples):
        cls = np.random.randint(0, 3)
        y[i] = cls
        
        if cls == 0:
            for t in range(seq_len):
                X[i, t] = np.random.randn(embedding_dim) * 0.1
        else:
            k = np.random.randint(1, seq_len) # escalation starts at turn k
            for t in range(seq_len):
                if t < k:
                    X[i, t] = np.random.randn(embedding_dim) * 0.1
                else:
                    direction = pair_direction if cls == 1 else crescendo_direction
                    strength = 0.5 + 0.1 * (t - k)
                    X[i, t] = direction * strength + np.random.randn(embedding_dim) * 0.05
                    
    # Normalize each turn's embedding vector to unit length
    for i in range(num_samples):
        for t in range(seq_len):
            norm = np.linalg.norm(X[i, t])
            if norm > 0:
                X[i, t] /= norm
                
    return X, y

def train_and_export():
    dummy_x, dummy_y = generate_synthetic_temporal_data(1000, 5, 384)
    
    dataset = TemporalSecurityDataset(dummy_x, dummy_y)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=True)
    
    device = "cpu"
    if torch.cuda.is_available():
        device = "cuda"
    elif torch.backends.mps.is_available():
        device = "mps"
        
    model = BiLSTMTemporalClassifier().to(device)
    criterion = FocalLoss(gamma=2.0)
    optimizer = optim.AdamW(model.parameters(), lr=1e-3)
    
    model.train()
    for epoch in range(3):
        for seqs, lbls in dataloader:
            seqs, lbls = seqs.to(device), lbls.to(device)
            optimizer.zero_grad()
            preds = model(seqs)
            loss = criterion(preds, lbls)
            loss.backward()
            optimizer.step()
            
    model.eval().cpu()
    dummy_input = torch.randn(1, 5, 384)
    
    os.makedirs("outputs", exist_ok=True)
    torch.onnx.export(
        model, dummy_input, "outputs/temporal_model.onnx",
        input_names=['embedding_sequence'], output_names=['class_logits'],
        opset_version=17
    )
    
    import onnx
    model_proto = onnx.load("outputs/temporal_model.onnx")
    onnx.save(model_proto, "outputs/temporal_model.onnx")
    
    data_file = "outputs/temporal_model.onnx.data"
    if os.path.exists(data_file):
        os.remove(data_file)
        
    print("[+] Temporal Model saved to outputs/temporal_model.onnx")

if __name__ == "__main__":
    train_and_export()