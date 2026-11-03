# Recurrent Sentiment Classification on the IMDb Dataset (Maas et al., 2011)

A clean, modular, and production-ready PyTorch codebase comparing **Vanilla RNN**, **LSTM**, **GRU**, and **Bidirectional LSTM (BiLSTM)** for sentiment analysis on the **IMDb Large Movie Review Dataset** (50,000 reviews: 25,000 train, 25,000 test).

---

## 📌 Theoretical Highlights

### 1. The Vanishing Gradient Problem (Vanilla RNN)
In an Elman RNN with hidden state $h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$, Backpropagation Through Time (BPTT) computes gradients across time steps via chained Jacobians:
$$\frac{\partial \mathcal{L}}{\partial h_k} = \frac{\partial \mathcal{L}}{\partial h_T} \prod_{j=k+1}^T \frac{\partial h_j}{\partial h_{j-1}}$$
Because $|\tanh'(z)| \le 1$, continuous matrix multiplication results in exponential gradient decay over longer sequences ($\sim \gamma^{T-k} \to 0$), impairing the model's ability to retain long-range context.

### 2. The LSTM Solution (Constant Error Carousel)
LSTMs mitigate vanishing gradients by introducing an additive internal **Cell State** ($C_t$):
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$
Because $\frac{\partial C_t}{\partial C_{t-1}} = f_t$, when the forget gate $f_t \approx 1.0$, the gradient flows back across time steps without exponential decay—forming the **Constant Error Carousel** (Hochreiter & Schmidhuber, 1997).

### 3. GRU & Bidirectional LSTM
- **GRU** couples forget and input gates into a unified update gate ($z_t$) and reset gate ($r_t$), reducing parameter count while preserving long-term gradient flow.
- **BiLSTM** captures bidirectional temporal context by concatenating forward ($\vec{h}_T$) and backward ($\overleftarrow{h}_1$) hidden representations.

---

## 📂 Repository Structure

```
Imdb_sentiment_analysis-using-Rnn/
│
├── data/                                 # Data folder
│   ├── raw/                              # Raw Stanford aclImdb archive & extracts (.gitkeep)
│   └── processed/                        # Parsed imdb_train.csv & imdb_test.csv (.gitkeep)
│
├── notebooks/                            # Exploratory Jupyter Notebooks
│   └── imdb_sentiment_analysis.ipynb     # Interactive research notebook
│
├── outputs/                              # Experiment artifacts
│   ├── figures/                          # Saved loss curves, confusion matrices, bar charts
│   └── models/                           # Trained PyTorch model checkpoints (.pt)
│
├── config.py                             # Deterministic seed (42), device selector, hyperparameter defaults
├── preprocessor.py                       # Contractions mapping, regex cleaner & tokenization
├── vocabulary.py                         # Strict anti-leakage vocabulary engine (pad=0, unk=1, min_freq=2)
├── dataset.py                            # IMDb data fetcher, PyTorch Dataset & DataLoader factory
├── models.py                             # VanillaRNN, LSTM, GRU, BiLSTM model definitions
├── engine.py                             # Universal training & evaluation loops (fit, evaluate, early stopping)
├── visualize.py                          # Publication-quality plotting routines
├── predict.py                            # Sentiment inference function & benchmark test reviews
│
├── main.py                               # Full end-to-end training & evaluation pipeline
├── tune.py                               # Hyperparameter tuning runner (Sequence Length & Hidden Dim)
├── inference.py                          # CLI & interactive sentiment predictor
│
├── requirements.txt                      # Project dependencies
├── .gitignore                            # Git exclusion rules
└── README.md                             # Documentation & theoretical breakdown
```

---

## ⚡ Installation

Clone the repository and install the dependencies:

```bash
git clone https://github.com/newone-ss/Imdb_sentiment_analysis-using-Rnn.git
cd Imdb_sentiment_analysis-using-Rnn
pip install -r requirements.txt
```

---

## 🚀 Usage

### 1. End-to-End Pipeline (`main.py`)
Trains all 4 recurrent architectures (Vanilla RNN, LSTM, GRU, BiLSTM) on the 20,000 training subset, validates on 5,000 reviews with early stopping, evaluates on the 25,000 test set, logs the benchmark summary table, saves figures to `outputs/figures/`, and runs sample predictions:

```bash
python main.py
```

Optional arguments:
```bash
python main.py --epochs 4 --batch_size 64 --seq_len 200 --lr 0.001 --hidden_dim 128
```

### 2. Hyperparameter Tuning (`tune.py`)
Runs empirical sweeps on:
- **Sequence Length ($L \in [100, 200, 300, 500]$)**
- **Hidden Dimension ($H \in [64, 128, 256]$)**

Generates sensitivity curves side-by-side in `outputs/figures/hyperparameter_tuning.png`:
```bash
python tune.py --epochs 3 --batch_size 64
```

### 3. Sentiment Inference (`inference.py`)
Run inference on single reviews or custom text using a trained model checkpoint:

```bash
# Predict single review
python inference.py --model_type bilstm --checkpoint outputs/models/bilstm.pt --text "An absolute masterpiece with breathtaking cinematography!"

# Run default benchmark test reviews
python inference.py --model_type bilstm --checkpoint outputs/models/bilstm.pt
```

---

## 🔬 Benchmark Comparison (IMDb 25,000 Test Set)

| Architecture | Hidden Dim | Dropout | Parameters | Loss Function | Optimizer |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Vanilla RNN** | 128 | 0.3 | ~3.23M | `BCEWithLogitsLoss` | `Adam (lr=1e-3)` |
| **LSTM** | 128 | 0.3 | ~3.36M | `BCEWithLogitsLoss` | `Adam (lr=1e-3)` |
| **GRU** | 128 | 0.3 | ~3.32M | `BCEWithLogitsLoss` | `Adam (lr=1e-3)` |
| **BiLSTM** | 128 | 0.3 | ~3.49M | `BCEWithLogitsLoss` | `Adam (lr=1e-3)` |

---

## 🛡️ Anti-Leakage Rules
- **Vocabulary Creation**: Built strictly from training set tokens; zero exposure to validation or test reviews.
- **Fixed Tokens**: `<pad>` mapped to index 0 (with zeroed embedding gradients), `<unk>` mapped to index 1.
- **Stratified Split**: 80/20 train/validation split stratified on binary sentiment labels (`random_state=42`).
- **Gradient Clipping**: Maximum norm threshold of `5.0` to avert exploding gradients during BPTT.
