# Amanu — Kapampangan Language Model Predictor

Amanu is a simple n-gram language model that predicts the next word in Kapampangan text. Give it a syllable or partial phrase, and it suggests the most likely next words based on probabilities learned from a training corpus.

"Amanu" is Kapampangan for "word/speech," and the name reflects the project's goal: helping explore and predict patterns in the Kapampangan language.

## Features

- N-gram modeling (configurable order, default trigram) with add-alpha (Laplace) smoothing
- Automatic backoff to shorter contexts when a longer n-gram hasn't been seen
- Softmax-based probability ranking of candidate next words
- Interactive command-line interface
- Simple feedback loop: incorrect predictions get added back into the training data for future runs

## How It Works

1. **Tokenization** — input text is lowercased and split into words and punctuation.
2. **N-gram counting** — the training corpus is processed into unigram, bigram, and trigram (or higher) frequency counts, with `<s>` / `</s>` markers for sentence boundaries.
3. **Conditional probability** — given a context (history of previous words), the model computes a smoothed probability for each candidate next word, backing off to shorter histories if needed.
4. **Prediction** — candidates are ranked by probability (via softmax) and the top results are shown.
5. **Feedback** — if the prediction is wrong, the next-best guesses are appended to the training file, helping the model improve over time.

## Requirements

- Python 3.8+
- No external dependencies (uses only the standard library: `re`, `math`, `collections`, `typing`)

## Setup

1. Clone this repository.
2. Prepare a training corpus: a plain text file with one Kapampangan sentence per line.
3. Update the file path in the script to point to your training file:

```python
with open(r'path/to/your/training.txt', 'r', encoding='utf-8') as f:
    corpus = [line.strip() for line in f if line.strip()]
```

   (Note: the current script uses a Windows-specific absolute path — update this to a relative path or your own file location before running.)

## Usage

Run the script:

```bash
python Amanu_Full_Explanation.py
```

You'll be greeted with an interactive prompt:

```
===== Welcome to Amanu: The Kapampangan Language Model Predictor! =====

Enter a syllable to predict the next word (or type 'exit' to quit):
```

Type a word or partial phrase, and Amanu will return its top predictions with probabilities:

```
🔹 Input Context:
   ['lub']

🔹 Predicted Next Words:
   lub (Probability: 0.4521)
   lugud (Probability: 0.2103)
   ...
```

After each prediction, you'll be asked whether it was correct. If not, alternative suggestions are saved to the training file to help refine future predictions.

Type `exit` to quit.

## Configuration

The model is instantiated with two key parameters:

```python
kapampanganLangModel = KapampanganLanguageModel(n=3, alpha=0.5)
```

- `n` — the n-gram order (3 = trigram model)
- `alpha` — smoothing parameter for unseen n-grams (higher = more smoothing)

## Project Status

This is a learning/exploratory project focused on the Kapampangan language. The current implementation centers on practical n-gram modeling with detailed inline explanations of each step — useful both as a working predictor and as an educational reference for how simple statistical language models work.

## Author

Crisiane Josef A. Caranto

## Acknowledgments

*Luid ing Amanung Kapampangan!*
