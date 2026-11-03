import os
import argparse
import torch

from config import get_device, DATA_ROOT, SEED, set_seed
from dataset import get_imdb_data
from preprocessor import TextPreprocessor
from vocabulary import Vocabulary
from models import BiLSTMClassifier, LSTMClassifier, GRUClassifier, VanillaRNN
from predict import predict_sentiment, TEST_REVIEWS

MODEL_MAP = {
    "bilstm": BiLSTMClassifier,
    "lstm": LSTMClassifier,
    "gru": GRUClassifier,
    "vanilla_rnn": VanillaRNN,
}


def parse_args():
    parser = argparse.ArgumentParser(description="Sentiment Inference with Recurrent Models")
    parser.add_argument("--model_type", type=str, default="bilstm", choices=["bilstm", "lstm", "gru", "vanilla_rnn"], help="Model architecture")
    parser.add_argument("--checkpoint", type=str, default="outputs/models/bilstm.pt", help="Path to model weights checkpoint")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Data root used to rebuild vocabulary")
    parser.add_argument("--text", type=str, default=None, help="Specific review text to classify")
    parser.add_argument("--seq_len", type=int, default=200, help="Max sequence length")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device()

    print(f"Loading vocabulary from {args.data_root}...")
    df_train, _ = get_imdb_data(args.data_root)
    preprocessor = TextPreprocessor()
    train_tokenized = [preprocessor.tokenize(t) for t in df_train["text"]]
    vocab = Vocabulary(min_freq=2, max_size=25000).build_vocab(train_tokenized)

    model_cls = MODEL_MAP[args.model_type]
    model = model_cls(len(vocab), embedding_dim=128, hidden_dim=128, dropout=0.3)

    if os.path.exists(args.checkpoint):
        print(f"Loading model checkpoint from {args.checkpoint}...")
        model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    else:
        print(f"Warning: Checkpoint '{args.checkpoint}' not found. Running with uninitialized weights.")

    model.to(device)

    if args.text:
        res = predict_sentiment(args.text, model, vocab, preprocessor, seq_len=args.seq_len, device=device)
        print("\n--- Prediction Result ---")
        print(f"Review:      {res['review']}")
        print(f"Sentiment:   {res['sentiment']}")
        print(f"Confidence:  {res['confidence']*100:.2f}%")
        print(f"Probability: {res['probability']:.4f}")
    else:
        print("\n" + "=" * 92)
        print("DEFAULT TEST REVIEWS DEMONSTRATION")
        print("=" * 92)
        print(f"{'Review Sample':<65} | {'Sentiment':<10} | {'Confidence':<10}")
        print("-" * 92)
        for r in TEST_REVIEWS:
            res = predict_sentiment(r, model, vocab, preprocessor, seq_len=args.seq_len, device=device)
            short_r = (r[:62] + "...") if len(r) > 62 else r
            print(f"{short_r:<65} | {res['sentiment']:<10} | {res['confidence']*100:>8.2f}%")


if __name__ == "__main__":
    main()
