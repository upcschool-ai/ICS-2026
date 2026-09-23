import logging
import os
import pathlib
import re

import torch
from flask import Flask, render_template, request

from model import SentimentAnalysis


def tokenize(text: str):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    return text.split()


app = Flask(__name__)

VOCAB_WORD2IDX = None
MODEL = None
NGRAMS = None


def load_model():
    global VOCAB_WORD2IDX, MODEL, NGRAMS

    checkpoint_path = pathlib.Path(__file__).parent.absolute() / "state_dict.pt"
    checkpoint = torch.load(checkpoint_path, weights_only=False)

    VOCAB_WORD2IDX = checkpoint["vocab_word2idx"]

    # TODO: Create the model and load the trained weights from the checkpoint.
    #
    # Hint:
    #   1. The model parameters are stored in the checkpoint:
    #        vocab_size = len(VOCAB_WORD2IDX)
    #        embed_dim  = checkpoint["embed_dim"]
    #        num_class  = checkpoint["num_class"]
    #   2. Create the model: SentimentAnalysis(...)
    #   3. Load the saved weights: model.load_state_dict(checkpoint["model_state_dict"])
    #   4. Set the model to evaluation mode: model.eval()
    MODEL = ...

    NGRAMS = checkpoint["ngrams"]


@torch.no_grad()
def predict_review_sentiment(text):
    tokens = tokenize(text)
    text_tensor = torch.tensor(
        [VOCAB_WORD2IDX.get(token, VOCAB_WORD2IDX["<unk>"]) for token in tokens]
    )

    if len(text_tensor) == 0:
        text_tensor = torch.tensor([VOCAB_WORD2IDX["<unk>"]])

    # TODO: Run the model on the text to get a prediction.
    #
    # Hint: read the model.py file, to understand inputs and outputs.
    output = ...

    confidences = torch.softmax(output, dim=1)
    return confidences.squeeze()[1].item()


@app.route("/predict", methods=["POST"])
def predict():
    """The input parameter is `review`"""
    review = request.form["review"]
    print(f"Prediction for review:\n {review}")

    result = predict_review_sentiment(review)
    return render_template("result.html", result=result)


@app.route("/", methods=["GET"])
def hello():
    """Return an HTML."""
    return render_template("hello.html")


@app.errorhandler(500)
def server_error(e):
    logging.exception('An error occurred during a request.')
    return f"""
    An internal error occurred: <pre>{e}</pre>
    See logs for full stacktrace.
    """, 500


load_model()

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)), debug=True)
