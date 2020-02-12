import re
from collections import Counter

import joblib
import jieba
from tqdm import tqdm


SPACE_NORMALIZER = re.compile(r"\s+")


def load_raw_data(path: str):
    print(f"Loading data from {path}...")
    raw_data = joblib.load(path)
    return raw_data


def save_raw_data(raw_data, path: str):
    print(f"Saving data to {path}...")
    joblib.dump(raw_data, path)


def char_tokenizer(text: str):
    return list(text)


def space_tokenizer(text: str):
    text = SPACE_NORMALIZER.sub(" ", text)
    return text.split()


def jieba_tokenizer(text: str):
    return jieba.lcut(text)


def create_tokenizer(method):
    if method == "char":
        return char_tokenizer
    elif method == "space":
        return space_tokenizer
    elif method == "jieba":
        return jieba_tokenizer
    else:
        supported_methods = ["char", "space", "jieba"]
        raise TypeError(f"tokenizer method {method} not in {supported_methods}!")


def tokenize_file(path, tokenizer):
    print(f"Tokenize text from {path}...")
    tokenized_pairs = []
    with open(path) as f:
        for line in tqdm(f):
            text, label = line.strip().split('\t')
            text = " ".join(tokenizer(text))
            tokenized_pairs.append((text, label))
    return tokenized_pairs


def get_label_prob(labels):
    counter = Counter(labels)
    size = len(labels)
    probs = {}
    for label in counter:
        probs[label] = counter[label]/size
    return probs


def build_label2id(labels):
    label2id = {}
    for label in labels:
        if label not in label2id:
            label2id[label] = len(label2id)
    return label2id
