import tarfile
import urllib.request
from pathlib import Path
from typing import List, Tuple
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split

from preprocessor import TextPreprocessor
from vocabulary import Vocabulary

# Dataset Downloader and Caching Routine
def get_imdb_data(data_root: str = "data") -> Tuple[pd.DataFrame, pd.DataFrame]:
    root_dir = Path(data_root)
    processed_dir = root_dir / "processed"
    raw_dir = root_dir / "raw"
    processed_dir.mkdir(parents=True, exist_ok=True)
    raw_dir.mkdir(parents=True, exist_ok=True)

    train_csv = processed_dir / "imdb_train.csv"
    test_csv = processed_dir / "imdb_test.csv"

    # 1. Use cached CSVs if available
    if train_csv.exists() and test_csv.exists():
        print(f"[Dataset] Loading cached CSVs from {processed_dir}...")
        df_train = pd.read_csv(train_csv)
        df_test = pd.read_csv(test_csv)
        return df_train, df_test

    # 2. Check if raw directory exists
    acl_dir = raw_dir / "aclImdb"
    archive_path = raw_dir / "aclImdb_v1.tar.gz"

    if not (acl_dir / "train" / "pos").exists():
        if not archive_path.exists():
            url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
            print(f"[Dataset] Downloading Maas et al. (2011) archive from {url}...")
            urllib.request.urlretrieve(url, archive_path)
            print("[Dataset] Download complete.")

        print(f"[Dataset] Extracting {archive_path}...")
        with tarfile.open(archive_path, "r:gz") as tar:
            tar.extractall(raw_dir)
        print("[Dataset] Extraction complete.")

    # 3. Fast multi-file parsing
    def parse_split(split_name: str) -> pd.DataFrame:
        data = []
        for sentiment, label in [("pos", 1), ("neg", 0)]:
            folder = acl_dir / split_name / sentiment
            for txt_file in folder.glob("*.txt"):
                try:
                    with open(txt_file, "r", encoding="utf-8") as f:
                        data.append({"text": f.read().strip(), "label": label})
                except Exception:
                    continue
        return pd.DataFrame(data)

    print("[Dataset] Parsing 25,000 training reviews (12.5k pos, 12.5k neg)...")
    df_train = parse_split("train")
    print("[Dataset] Parsing 25,000 test reviews (12.5k pos, 12.5k neg)...")
    df_test = parse_split("test")

    df_train.to_csv(train_csv, index=False)
    df_test.to_csv(test_csv, index=False)
    print(f"[Dataset] Successfully saved cached CSVs to {processed_dir}")
    return df_train, df_test


# PyTorch IMDb Dataset
class IMDbDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, preprocessor: TextPreprocessor, max_seq_length: int = 200):
        self.labels = labels
        self.max_seq_length = max_seq_length
        self.encoded_samples = []

        for t in texts:
            tokens = preprocessor.tokenize(t)
            ids = vocab.encode(tokens)
            self.encoded_samples.append(ids)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        ids = self.encoded_samples[idx]
        seq_len = len(ids)

        if seq_len >= self.max_seq_length:
            padded = ids[:self.max_seq_length]
            actual_len = self.max_seq_length
        else:
            padded = ids + [Vocabulary.PAD_IDX] * (self.max_seq_length - seq_len)
            actual_len = seq_len

        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "length": torch.tensor(actual_len, dtype=torch.long),
            "label": torch.tensor(self.labels[idx], dtype=torch.float32),
        }


def create_dataloaders(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    vocab: Vocabulary,
    preprocessor: TextPreprocessor,
    seq_len: int = 200,
    batch_size: int = 64,
    val_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    # Stratified Train/Val Split (80% train = 20k, 20% val = 5k)
    train_sub_df, val_df = train_test_split(
        df_train, test_size=val_size, random_state=random_state, stratify=df_train["label"]
    )

    train_ds = IMDbDataset(train_sub_df["text"].tolist(), train_sub_df["label"].tolist(), vocab, preprocessor, seq_len)
    val_ds = IMDbDataset(val_df["text"].tolist(), val_df["label"].tolist(), vocab, preprocessor, seq_len)
    test_ds = IMDbDataset(df_test["text"].tolist(), df_test["label"].tolist(), vocab, preprocessor, seq_len)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    return train_loader, val_loader, test_loader

