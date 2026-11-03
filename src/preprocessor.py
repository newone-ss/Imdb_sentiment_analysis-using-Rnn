import re
from typing import List

# Preprocessing & Tokenization Module
CONTRACTIONS = {
    r"won't": "will not",
    r"can't": "cannot",
    r"n't": " not",
    r"'re": " are",
    r"'s": " is",
    r"'d": " would",
    r"'ll": " will",
    r"'t": " not",
    r"'ve": " have",
    r"'m": " am",
}

class TextPreprocessor:
    def __init__(self, lower: bool = True, remove_html: bool = True, expand_contractions: bool = True):
        self.lower = lower
        self.remove_html = remove_html
        self.expand_contractions = expand_contractions

    def clean_text(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        if self.remove_html:
            text = re.sub(r"<[^>]+>", " ", text)
        if self.lower:
            text = text.lower()
        if self.expand_contractions:
            for pattern, repl in CONTRACTIONS.items():
                text = re.sub(pattern, repl, text)
        text = re.sub(r"[^a-zA-Z\s]", " ", text)
        text = re.sub(r"\s+", " ", text).strip()
        return text

    def tokenize(self, text: str) -> List[str]:
        return self.clean_text(text).split()

