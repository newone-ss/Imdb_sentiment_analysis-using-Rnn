import torch
import torch.nn as nn
from typing import Dict, Any

from config import get_device
from preprocessor import TextPreprocessor
from vocabulary import Vocabulary

# Standalone Prediction Function
def predict_sentiment(
    review_text: str,
    model: nn.Module,
    vocab: Vocabulary,
    preprocessor: TextPreprocessor,
    seq_len: int = 200,
    device: torch.device = None
) -> Dict[str, Any]:
    if device is None:
        device = get_device()

    model.eval()
    tokens = preprocessor.tokenize(review_text)
    ids = vocab.encode(tokens)

    if len(ids) >= seq_len:
        padded = ids[:seq_len]
    else:
        padded = ids + [Vocabulary.PAD_IDX] * (seq_len - len(ids))

    tensor = torch.tensor([padded], dtype=torch.long, device=device)
    with torch.no_grad():
        prob = torch.sigmoid(model(tensor)).item()

    sentiment = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else (1.0 - prob)

    return {
        "review": review_text,
        "sentiment": sentiment,
        "confidence": confidence,
        "probability": prob
    }

# Default Benchmark Test Reviews (Cell 24)
TEST_REVIEWS = [
    "An absolute masterpiece! The director and cast delivered an unforgettable emotional experience.",
    "Completely unwatchable. Horrible dialogue, wooden performances, and a nonsensical plot.",
    "The cinematography was breathtaking and visually stunning, but the pacing dragged considerably in the middle.",
    "I expected this to be terrible given the reviews, but I was pleasantly surprised by the witty humor and charm!"
]

