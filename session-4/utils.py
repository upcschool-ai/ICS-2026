import os
import torch
from torch.utils.data import Dataset
from typing import List, Tuple
from collections import Counter
import re


class Vocabulary:
    def __init__(self, texts: List[str], min_freq: int = 1):
        self.word2idx = {"<pad>": 0, "<unk>": 1}
        self.idx2word = ["<pad>", "<unk>"]

        word_freq = Counter()
        for text in texts:
            words = self._tokenize(text)
            word_freq.update(words)

        for word, freq in word_freq.items():
            if freq >= min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word.append(word)

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower()
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        return text.split()

    def __len__(self):
        return len(self.word2idx)

    def encode(self, text: str) -> List[int]:
        words = self._tokenize(text)
        return [self.word2idx.get(word, self.word2idx["<unk>"]) for word in words]


class YelpDataset(Dataset):
    def __init__(self, texts: List[str], labels: List[int], vocab: Vocabulary, device=None):
        self.texts = texts
        self.labels = labels
        self.vocab = vocab
        self.device = device or torch.device('cpu')

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx) -> Tuple[torch.Tensor, torch.Tensor]:
        text = self.texts[idx]
        label = self.labels[idx]
        encoded = self.vocab.encode(text)
        return (torch.tensor(encoded, dtype=torch.long, device=self.device),
                torch.tensor(label, dtype=torch.long, device=self.device))


def collate_batch(batch):
    texts, labels = zip(*batch)
    lengths = [len(text) for text in texts]
    offsets = torch.tensor([0] + lengths[:-1], device=texts[0].device).cumsum(0)
    texts = torch.cat(texts)
    labels = torch.stack(labels)
    return texts, offsets, labels


class YelpReviewPolarityDatasetLoader:
    def __init__(self, ngrams: int = 1, batch_size: int = 16, device=None, data_dir="yelp_review_polarity_csv"):
        self.batch_size = batch_size
        self.data_dir = data_dir
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self._load_datasets()
        self.vocab = Vocabulary(self.train_texts)

    def _load_datasets(self):
        def read_data(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                lines = f.readlines()[1:]
                texts, labels = [], []
                for line in lines:
                    parts = line.split('","')
                    if len(parts) >= 2:
                        label = int(parts[0].strip('"'))
                        text = parts[1].strip().strip('"')
                        texts.append(text)
                        labels.append(label - 1)
                return texts, labels

        self.train_texts, self.train_labels = read_data(os.path.join(self.data_dir, 'train.csv'))
        self.test_texts, self.test_labels = read_data(os.path.join(self.data_dir, 'test.csv'))

    def get_vocab_size(self):
        return len(self.vocab)

    def get_num_classes(self):
        return 2

    def get_train_val_dataset(self):
        return YelpDataset(self.train_texts, self.train_labels, self.vocab, self.device)

    def get_test_dataset(self):
        return YelpDataset(self.test_texts, self.test_labels, self.vocab, self.device)

    def generate_batch(self, batch):
        return collate_batch(batch)
