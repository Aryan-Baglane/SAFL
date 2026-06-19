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

def train_and_export():
    dummy_x = np.random.randn(1000, 5, 384) # Sequence window size of 5 turns
    dummy_y = np.random.randint(0, 3, 1000)   # Classes: 0: SAFE, 1: PAIR, 2: CRESCENDO
    
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