import random
import numpy as np
import torch

# Deterministic Seeding Function (Rules Compliance)
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Default Hyperparameters & Paths
DATA_ROOT = "data"
SEED = 42
EMBEDDING_DIM = 128
HIDDEN_DIM = 128
DROPOUT = 0.3
MAX_SEQ_LENGTH = 200
BATCH_SIZE = 64
EPOCHS = 4
LR = 0.001
WEIGHT_DECAY = 1e-5
CLIP_NORM = 5.0
PATIENCE = 2
VOCAB_MIN_FREQ = 2
VOCAB_MAX_SIZE = 25000

