from typing import List
from collections import Counter

# Anti-Leakage Vocabulary Class
class Vocabulary:
    PAD_TOKEN = "<pad>"
    UNK_TOKEN = "<unk>"
    PAD_IDX = 0
    UNK_IDX = 1

    def __init__(self, min_freq: int = 2, max_size: int = 25000):
        self.min_freq = min_freq
        self.max_size = max_size
        self.word2idx = {self.PAD_TOKEN: self.PAD_IDX, self.UNK_TOKEN: self.UNK_IDX}
        self.idx2word = {self.PAD_IDX: self.PAD_TOKEN, self.UNK_IDX: self.UNK_TOKEN}

    def build_vocab(self, tokenized_texts: List[List[str]]):
        counter = Counter()
        for tokens in tokenized_texts:
            counter.update(tokens)

        filtered = [w for w, c in counter.items() if c >= self.min_freq]
        filtered.sort(key=lambda w: (-counter[w], w))

        capacity = self.max_size - 2
        filtered = filtered[:capacity]

        for w in filtered:
            idx = len(self.word2idx)
            self.word2idx[w] = idx
            self.idx2word[idx] = w

        print(f"[Vocabulary] Built vocabulary with {len(self.word2idx):,} tokens (pad={self.PAD_IDX}, unk={self.UNK_IDX}).")
        return self

    def __len__(self):
        return len(self.word2idx)

    def encode(self, tokens: List[str]) -> List[int]:
        unk = self.UNK_IDX
        return [self.word2idx.get(t, unk) for t in tokens]

    def decode(self, indices: List[int]) -> List[str]:
        return [self.idx2word.get(i, self.UNK_TOKEN) for i in indices]

