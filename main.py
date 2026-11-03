import os
import argparse
import pandas as pd
import torch
import torch.nn as nn

from config import set_seed, get_device, DATA_ROOT, SEED
from dataset import get_imdb_data, create_dataloaders
from preprocessor import TextPreprocessor
from vocabulary import Vocabulary
from models import (
    VanillaRNN,
    LSTMClassifier,
    GRUClassifier,
    BiLSTMClassifier,
)
from engine import fit, evaluate
from visualize import (
    plot_loss_curves,
    plot_confusion_matrices,
    plot_model_comparison,
)
from predict import predict_sentiment, TEST_REVIEWS


def parse_args():
    parser = argparse.ArgumentParser(description="Recurrent Sentiment Classification on IMDb Dataset")
    parser.add_argument("--data_root", type=str, default=DATA_ROOT, help="Directory to store dataset")
    parser.add_argument("--seed", type=int, default=SEED, help="Random seed for reproducibility")
    parser.add_argument("--epochs", type=int, default=4, help="Number of training epochs")
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for DataLoader")
    parser.add_argument("--seq_len", type=int, default=200, help="Maximum sequence length")
    parser.add_argument("--embedding_dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--hidden_dim", type=int, default=128, help="Recurrent hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout probability")
    parser.add_argument("--patience", type=int, default=2, help="Patience for early stopping")
    parser.add_argument("--output_dir", type=str, default="outputs", help="Directory for output artifacts")
    return parser.parse_args()


def main():
    args = parse_args()

    # 1. Deterministic Seeding & Device Setup
    set_seed(args.seed)
    device = get_device()
    print("=" * 55)
    print(f"PyTorch Version: {torch.__version__}")
    print(f"Device:          {device}")
    if torch.cuda.is_available():
        print(f"GPU Accelerator: {torch.cuda.get_device_name(0)}")
    else:
        print("Running on CPU.")
    print("=" * 55)

    os.makedirs(os.path.join(args.output_dir, "figures"), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "models"), exist_ok=True)

    # 2. Dataset Acquisition
    df_train, df_test = get_imdb_data(args.data_root)
    print(f"Train Dataset Shape: {df_train.shape} | Test Dataset Shape: {df_test.shape}")
    print("\nClass Distribution in Train:")
    print(df_train["label"].value_counts())

    # 3. Preprocessing & Vocabulary Construction (Anti-Leakage)
    preprocessor = TextPreprocessor()
    print("\nTokenizing training set...")
    train_tokenized = [preprocessor.tokenize(t) for t in df_train["text"]]
    vocab = Vocabulary(min_freq=2, max_size=25000).build_vocab(train_tokenized)

    # 4. PyTorch DataLoaders
    train_loader, val_loader, test_loader = create_dataloaders(
        df_train=df_train,
        df_test=df_test,
        vocab=vocab,
        preprocessor=preprocessor,
        seq_len=args.seq_len,
        batch_size=args.batch_size,
    )
    batch = next(iter(train_loader))
    print(f"Batch input_ids shape: {batch['input_ids'].shape} (batch_size, max_seq_length)")
    print(f"Batch labels shape:    {batch['label'].shape}")

    # 5. Model Architectures Zoo & Parameter Inspection
    models_zoo = {
        "Vanilla RNN": VanillaRNN(len(vocab), embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout=args.dropout),
        "LSTM":        LSTMClassifier(len(vocab), embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout=args.dropout),
        "GRU":         GRUClassifier(len(vocab), embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout=args.dropout),
        "BiLSTM":      BiLSTMClassifier(len(vocab), embedding_dim=args.embedding_dim, hidden_dim=args.hidden_dim, dropout=args.dropout),
    }

    print("\n" + f"{'Architecture':<18} | {'Trainable Parameters':<22}")
    print("-" * 44)
    for name, m in models_zoo.items():
        print(f"{name:<18} | {m.count_parameters():>20,}")

    # 6. Training & Comparative Evaluation
    trained_models = {}
    histories = {}
    test_results = {}
    criterion = nn.BCEWithLogitsLoss()

    for name, model in models_zoo.items():
        print(f"\n{'='*55}\n  Training {name.upper()}\n{'='*55}")
        hist = fit(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=args.epochs,
            lr=args.lr,
            patience=args.patience,
            device=device,
        )

        # Evaluate on the 25,000 Unseen Test Reviews
        test_metrics = evaluate(model, test_loader, criterion, device=device)

        trained_models[name] = model
        histories[name] = hist
        test_results[name] = test_metrics

        # Save model checkpoint
        ckpt_path = os.path.join(args.output_dir, "models", f"{name.lower().replace(' ', '_')}.pt")
        torch.save(model.state_dict(), ckpt_path)

        print(f"--> [{name}] Unseen Test Accuracy: {test_metrics['accuracy']*100:.2f}% | Macro F1: {test_metrics['f1_macro']*100:.2f}%")

    # 7. Benchmark Summary Table
    summary_data = []
    for name, m in test_results.items():
        summary_data.append({
            "Model": name,
            "Accuracy (%)": round(m["accuracy"] * 100, 2),
            "Macro F1 (%)": round(m["f1_macro"] * 100, 2),
            "Weighted F1 (%)": round(m["f1_weighted"] * 100, 2),
            "Precision (%)": round(m["precision"] * 100, 2),
            "Recall (%)": round(m["recall"] * 100, 2),
            "Parameters": f"{models_zoo[name].count_parameters():,}"
        })

    df_benchmark = pd.DataFrame(summary_data)
    print("\n" + "=" * 55)
    print("ARCHITECTURAL BENCHMARK SUMMARY TABLE")
    print("=" * 55)
    print(df_benchmark.to_string(index=False))

    summary_csv_path = os.path.join(args.output_dir, "benchmark_summary.csv")
    df_benchmark.to_csv(summary_csv_path, index=False)
    print(f"\n[Artifact] Benchmark summary saved to {summary_csv_path}")

    # 8. Visualizations
    fig_dir = os.path.join(args.output_dir, "figures")
    plot_loss_curves(histories, save_path=os.path.join(fig_dir, "learning_curves.png"))
    plot_confusion_matrices(test_results, save_path=os.path.join(fig_dir, "confusion_matrices.png"))
    plot_model_comparison(test_results, save_path=os.path.join(fig_dir, "model_comparison.png"))

    # 9. Interactive Predictor Demo
    print("\n" + "=" * 92)
    print("INTERACTIVE SENTIMENT PREDICTOR DEMONSTRATION (BiLSTM)")
    print("=" * 92)
    print(f"{'Review Sample':<65} | {'Sentiment':<10} | {'Confidence':<10}")
    print("-" * 92)
    bilstm_model = trained_models["BiLSTM"]
    for r in TEST_REVIEWS:
        res = predict_sentiment(
            review_text=r,
            model=bilstm_model,
            vocab=vocab,
            preprocessor=preprocessor,
            seq_len=args.seq_len,
            device=device,
        )
        short_r = (r[:62] + "...") if len(r) > 62 else r
        print(f"{short_r:<65} | {res['sentiment']:<10} | {res['confidence']*100:>8.2f}%")


if __name__ == "__main__":
    main()
