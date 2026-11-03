import time
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from config import get_device

# Training & Evaluation Engine
def train_single_epoch(model, loader, optimizer, criterion, clip_norm=5.0, device=None):
    if device is None:
        device = get_device()
    model.train()
    total_loss, all_preds, all_labels = 0.0, [], []

    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["label"].to(device)

        optimizer.zero_grad()
        logits = model(x)
        loss = criterion(logits, y)
        loss.backward()

        if clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_norm)

        optimizer.step()
        total_loss += loss.item() * len(y)

        preds = (torch.sigmoid(logits).detach().cpu().numpy() >= 0.5).astype(int)
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy().astype(int))

    avg_loss = total_loss / len(loader.dataset)
    acc = accuracy_score(all_labels, all_preds)
    return avg_loss, acc


@torch.no_grad()
def evaluate(model, loader, criterion, device=None):
    if device is None:
        device = get_device()
    model.eval()
    total_loss, all_preds, all_labels, all_probs = 0.0, [], [], []

    for batch in loader:
        x = batch["input_ids"].to(device)
        y = batch["label"].to(device)

        logits = model(x)
        loss = criterion(logits, y)
        total_loss += loss.item() * len(y)

        probs = torch.sigmoid(logits).cpu().numpy()
        preds = (probs >= 0.5).astype(int)

        all_probs.extend(probs)
        all_preds.extend(preds)
        all_labels.extend(y.cpu().numpy().astype(int))

    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)

    return {
        "loss": total_loss / len(loader.dataset),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1_macro": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred),
        "probabilities": np.array(all_probs),
        "y_true": y_true,
        "y_pred": y_pred,
    }


def fit(model, train_loader, val_loader, epochs=4, lr=0.001, patience=2, device=None):
    if device is None:
        device = get_device()
    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()

    history = {"train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [], "val_f1": []}
    best_f1, best_weights = -1.0, None
    no_improve = 0

    for epoch in range(1, epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = train_single_epoch(model, train_loader, optimizer, criterion, device=device)
        v_metrics = evaluate(model, val_loader, criterion, device=device)

        history["train_loss"].append(tr_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(v_metrics["loss"])
        history["val_acc"].append(v_metrics["accuracy"])
        history["val_f1"].append(v_metrics["f1_macro"])

        elapsed = time.time() - t0
        print(f"Epoch [{epoch:02d}/{epochs:02d}] "
              f"Train Loss: {tr_loss:.4f} | Acc: {tr_acc*100:.2f}% || "
              f"Val Loss: {v_metrics['loss']:.4f} | Acc: {v_metrics['accuracy']*100:.2f}% | F1: {v_metrics['f1_macro']*100:.2f}% "
              f"({elapsed:.1f}s)")

        if v_metrics["f1_macro"] > best_f1:
            best_f1 = v_metrics["f1_macro"]
            best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"Early stopping triggered at epoch {epoch}.")
                break

    if best_weights:
        model.load_state_dict(best_weights)
    return history

