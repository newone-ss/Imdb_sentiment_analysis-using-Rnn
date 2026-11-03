import os
import argparse
import pandas as pd
import torch
import torch.nn as nn

from config import set_seed, get_device, DATA_ROOT, SEED
from dataset import get_imdb_data, create_dataloaders
from preprocessor import TextPreprocessor
from vocabulary import Vocabulary
from models import LSTMClassifier
from engine import fit, evaluate
from visualize import plot_tuning_sensitivity


def parse_args():
    parser = argparse.ArgumentParser(description="Hyperparameter Tuning for Recurrent Sentiment Analysis")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Directory to store dataset")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=3, help="Epochs per tuning experiment")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for output artifacts")
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()
    print("=" * 55)
    print("HYPERPARAMETER TUNING: Sequence Length & Hidden Dimension")
    print(f"Device: {device}")
    print("=" * 55)

    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)

    df_train, df_test = get_imdb_data(args.data_root)
    preprocessor = TextPreprocessor()
    train_tokenized = [preprocessor.tokenize(t) for t in df_train["text"]]
    vocab = Vocabulary(min_freq=2, max_size=25000).build_vocab(train_tokenized)
    criterion = nn.BCEWithLogitsLoss()

    # 1. Tuning Sequence Length L in [100, 200, 300, 500] on LSTM
    seq_lengths = [100, 200, 300, 500]
    seq_tuning_results = {}

    print("\n--- 1. Tuning Sequence Length (L) on LSTM ---")
    for sl in seq_lengths:
        tr_l, v_l, ts_l = create_dataloaders(
            df_train=df_train,
            df_test=df_test,
            vocab=vocab,
            preprocessor=preprocessor,
            seq_len=sl,
            batch_size=args.batch_size,
        )
        tune_lstm = LSTMClassifier(len(vocab), embedding_dim=128, hidden_dim=128)
        _ = fit(tune_lstm, tr_l, v_l, epochs=args.epochs, lr=args.lr, device=device)
        m = evaluate(tune_lstm, ts_l, criterion, device=device)
        seq_tuning_results[sl] = {
            "accuracy": m["accuracy"] * 100,
            "f1_macro": m["f1_macro"] * 100
        }
        print(f"L = {sl:3d} -> Test Acc: {m['accuracy']*100:.2f}% | Test F1: {m['f1_macro']*100:.2f}%")

    # 2. Tuning Hidden Dimension H in [64, 128, 256] on LSTM
    hidden_dims = [64, 128, 256]
    hidden_tuning_results = {}
    tr_l200, v_l200, ts_l200 = create_dataloaders(
        df_train=df_train,
        df_test=df_test,
        vocab=vocab,
        preprocessor=preprocessor,
        seq_len=200,
        batch_size=args.batch_size,
    )

    print("\n--- 2. Tuning Hidden Dimension (H) on LSTM ---")
    for hd in hidden_dims:
        tune_lstm_h = LSTMClassifier(len(vocab), embedding_dim=128, hidden_dim=hd)
        _ = fit(tune_lstm_h, tr_l200, v_l200, epochs=args.epochs, lr=args.lr, device=device)
        m = evaluate(tune_lstm_h, ts_l200, criterion, device=device)
        hidden_tuning_results[hd] = {
            "accuracy": m["accuracy"] * 100,
            "f1_macro": m["f1_macro"] * 100
        }
        print(f"H = {hd:3d} -> Test Acc: {m['accuracy']*100:.2f}% | Test F1: {m['f1_macro']*100:.2f}%")

    # Save sensitivity plot
    plot_path = os.path.join(args.output_dir, "figures", "hyperparameter_tuning.png")
    plot_tuning_sensitivity(
        seq_lengths=seq_lengths,
        seq_tuning_results=seq_tuning_results,
        hidden_dims=hidden_dims,
        hidden_tuning_results=hidden_tuning_results,
        save_path=plot_path
    )


if __name__ == "__main__":
    main()
