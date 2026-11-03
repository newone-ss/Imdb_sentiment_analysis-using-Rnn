import torch
import torch.nn as nn

# Base Classifier Module
class BaseRecurrentClassifier(nn.Module):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.dropout = nn.Dropout(dropout)
        self.hidden_dim = hidden_dim

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# 1. Vanilla RNN (Elman RNN)
class VanillaRNN(BaseRecurrentClassifier):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__(vocab_size, embedding_dim, hidden_dim, dropout)
        self.rnn = nn.RNN(embedding_dim, hidden_dim, batch_first=True, nonlinearity="tanh")
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        emb = self.dropout(self.embedding(input_ids))
        _, hn = self.rnn(emb)
        return self.fc(self.dropout(hn[-1])).squeeze(-1)


# 2. LSTM (Long Short-Term Memory)
class LSTMClassifier(BaseRecurrentClassifier):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__(vocab_size, embedding_dim, hidden_dim, dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        emb = self.dropout(self.embedding(input_ids))
        _, (hn, _) = self.lstm(emb)
        return self.fc(self.dropout(hn[-1])).squeeze(-1)


# 3. GRU (Gated Recurrent Unit)
class GRUClassifier(BaseRecurrentClassifier):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__(vocab_size, embedding_dim, hidden_dim, dropout)
        self.gru = nn.GRU(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)

    def forward(self, input_ids):
        emb = self.dropout(self.embedding(input_ids))
        _, hn = self.gru(emb)
        return self.fc(self.dropout(hn[-1])).squeeze(-1)


# 4. Bidirectional LSTM (BiLSTM)
class BiLSTMClassifier(BaseRecurrentClassifier):
    def __init__(self, vocab_size: int, embedding_dim: int = 128, hidden_dim: int = 128, dropout: float = 0.3):
        super().__init__(vocab_size, embedding_dim, hidden_dim, dropout)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, 1)

    def forward(self, input_ids):
        emb = self.dropout(self.embedding(input_ids))
        _, (hn, _) = self.lstm(emb)
        # Forward final hidden state is hn[-2], backward final hidden state is hn[-1]
        cat_hidden = torch.cat((hn[-2], hn[-1]), dim=1)
        return self.fc(self.dropout(cat_hidden)).squeeze(-1)

