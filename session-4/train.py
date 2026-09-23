import pathlib
import time
import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import random_split
from app.model import SentimentAnalysis
from utils import YelpReviewPolarityDatasetLoader
import torch.optim as optim
import torch.nn.functional as F
import numpy as np


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Paths relative to this file, so the script works no matter where you run it from
SESSION_DIR = pathlib.Path(__file__).resolve().parent
DATA_DIR = SESSION_DIR.parent / "yelp_review_polarity_csv"
CHECKPOINT_PATH = SESSION_DIR / "app" / "state_dict.pt"


def train(dataloader, model, optimizer, criterion):
    model.train()

    train_loss = 0
    train_acc = 0
    for text, offsets, label in dataloader:
        optimizer.zero_grad()
        output = model(text, offsets)
        loss = criterion(output, label)
        optimizer.step()

        train_loss += loss.item() * len(output)
        train_acc += (output.argmax(1) == label).sum().item()

    scheduler.step()

    return train_loss / len(dataloader.dataset), train_acc / len(dataloader.dataset)


def test(dataloader, model, criterion):
    model.eval()

    loss = 0
    acc = 0
    with torch.no_grad():
        for text, offsets, label in dataloader:
            output = model(text, offsets)
            loss_val = criterion(output, label)

            loss += loss_val.item() * len(output)
            acc += (output.argmax(1) == label).sum().item()

    return loss / len(dataloader.dataset), acc / len(dataloader.dataset)


if __name__ == "__main__":

    NGRAMS = 1
    BATCH_SIZE = 16
    EMBED_DIM = 32
    N_EPOCHS = 2

    yelp_loader = YelpReviewPolarityDatasetLoader(NGRAMS, BATCH_SIZE, device=device, data_dir=DATA_DIR)

    train_val_dataset = yelp_loader.get_train_val_dataset()
    test_dataset = yelp_loader.get_test_dataset()

    VOCAB_SIZE = yelp_loader.get_vocab_size()
    NUM_CLASS = yelp_loader.get_num_classes()

    model = SentimentAnalysis(VOCAB_SIZE, EMBED_DIM, NUM_CLASS).to(device)

    criterion = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.SGD(model.parameters(), lr=4.0)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 1, gamma=0.9)

    train_size = int(len(train_val_dataset) * 0.95)
    valid_size = int(len(train_val_dataset) * 0.05)
    train_dataset, valid_dataset = random_split(train_val_dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=yelp_loader.generate_batch)
    val_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yelp_loader.generate_batch)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=yelp_loader.generate_batch)

    for epoch in range(N_EPOCHS):

        start_time = time.time()
        train_loss, train_acc = train(train_loader, model, optimizer, criterion)
        valid_loss, valid_acc = test(val_loader, model, criterion)

        secs = int(time.time() - start_time)
        mins = secs / 60
        secs = secs % 60

        print(f"Epoch: {epoch + 1},  | time in {mins} minutes, {secs} seconds")
        print(f'\tLoss: {train_loss:.4f}(train)\t|\tAcc: {train_acc * 100:.1f}%(train)')
        print(f'\tLoss: {valid_loss:.4f}(valid)\t|\tAcc: {valid_acc * 100:.1f}%(valid)')

    print("Training finished")

    test_loss, test_acc = test(test_loader, model, criterion)
    print(f'\tLoss: {test_loss:.4f}(test)\t|\tAcc: {test_acc * 100:.1f}%(test)')

    print(f"Saving checkpoint to {CHECKPOINT_PATH}...")
    checkpoint = {
        "model_state_dict": model.cpu().state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "vocab_word2idx": yelp_loader.vocab.word2idx,
        "vocab_idx2word": yelp_loader.vocab.idx2word,
        "ngrams": NGRAMS,
        "embed_dim": EMBED_DIM,
        "num_class": NUM_CLASS,
    }
    torch.save(checkpoint, CHECKPOINT_PATH)
