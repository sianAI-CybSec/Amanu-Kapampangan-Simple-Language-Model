#Caranto, Crisiane Josef A.
#Kapampangan Simple Language Model Predictor

import re, math
from collections import defaultdict, Counter
from typing import List, Tuple, Dict

"""
This program is a simple language model predictor for the Kapampangan language. It uses a n-gram model to predict the next word in a given sentence. The program takes a corpus of text as input, 
processes it to create a bigram frequency dictionary, and then uses this dictionary to make predictions based on user input.

Flowchart:
    1. Start
    2. User Input
    3. Tokenization
    4. N-Gram Notation
    5. Conditional Probability Calculation
    6. Prediction
    7. Show Results
    8. Ask for Feedback (Correct Prediction?)
    9. Update Training Data (If Incorrect)
    10. Repeat or Exit

However, in this implementation, we will focus on the bigram model for simplicity. The program will read a corpus of Kapampangan text, create a bigram frequency dictionary, and then allow the user to input a sentence to predict the next word based on the bigram probabilities.
"""

BOS = "<s>"
EOS = "</s>" 

def tokenize(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9']+|[.,!?;:]", text.lower())

def make_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

class KapampanganLanguageModel:
    def __init__(self, n: int = 1, alpha: float = 0.5):
        self.n = n
        self.alpha = alpha
        self.vocab = set() 
        self.ngram_counts: Dict[int, Counter] = defaultdict(Counter) 

    def fit(self, corpus: List[str]) -> None:
        for line in corpus:
            toks = [BOS] * (self.n-1) + tokenize(line) + [EOS]
            self.vocab.update(toks) 
            for k in range(1, self.n+1): 
                for ng in make_ngrams(toks, k):
                    self.ngram_counts[k][ng] += 1
        self.Vocabulary = len(self.vocab) 

    def _conditional_logprob(self, word: str, history: Tuple[str, ...]) -> float:
        k = len(history) + 1 
        ngram = history + (word,)
        num = self.ngram_counts[k][ngram] + self.alpha
        if k == 1:
            totalUnigrams = sum(self.ngram_counts[1].values())
            den = totalUnigrams + self.alpha * self.Vocabulary
        else:
            histCount = self.ngram_counts[k-1][history]
            den = histCount + self.alpha * self.Vocabulary
        return math.log(num) - math.log(den)

    def next_word_probability(self, context: List[str]) -> List[Tuple[str, float]]:
        hist = [BOS] * max(0, self.n-1 - len(context)) + context[-(self.n-1):]
        #SOFTMAX FUNCTION
        for hlen in range(self.n - 1, -1, -1):
            history = tuple(hist[-hlen:] if hlen > 0 else ())
            if hlen == 0 or (hlen in self.ngram_counts and self.ngram_counts[hlen].get(history, 0) > 0):
                logps = [(w, self._conditional_logprob(w, history)) for w in self.vocab]
                m = max(lp for _, lp in logps)
                exps = [(w, math.exp(lp - m)) for w, lp in logps]
                Z = sum(v for _, v in exps)
                return [(w, v / Z) for w, v in exps]
                
            uni = 1.0 / self.Vocabulary 
            return [(w, uni) for w in self.vocab]
        
    def predict_next(self, context: List[str], top_k: int = 5, exclude_punct: bool = True) -> List[Tuple[str, float]]:
        probs = self.next_word_probability(context)
        if exclude_punct:
            probs = [(w, p) for w, p in probs if w not in {".", ",", "!", "?", ";", ":"}]
        probs.sort(key=lambda x: x[1], reverse=True)
        return probs[:top_k]
    
if __name__ == "__main__":
    with open(r'C:\Users\PYTHON PROGRAMS\AMANU\training.txt', 'r', encoding='utf-8') as f:
        corpus = [line.strip() for line in f if line.strip()]
    kapampanganLangModel = KapampanganLanguageModel(n=3, alpha=0.5)
    kapampanganLangModel.fit(corpus)

#MAIN Interface
#=====================================================================================
print("===== Welcome to Amanu: The Kapampangan Language Model Predictor! =====\n")
print("Amanu is a simple language model predictor designed to help you explore the Kapampangan language. It uses a n-gram model to predict the next word based on your input.")
print("You can enter a syllable or a partial word, and Amanu will provide you with the most likely next words based on the training data it has learned from.\n")

while True:
    alphaSyllable = input("Enter a syllable to predict the next word (or type 'exit' to quit): ")
    if alphaSyllable.lower() == 'exit':
        print("\nThank you for using Amanu! Goodbye!")
        print("Luid ing Amanung Kapampangan!")
        break

    ctx = tokenize(alphaSyllable)
    predictions = kapampanganLangModel.predict_next(ctx)

    print("\n🔹 Input Context:")
    print(f"   {ctx}\n")

    print("🔹 Predicted Next Words:")
    for word, prob in predictions:
        print(f"   {word} (Probability: {prob:.4f})")

    print("\nResult: {ctx} -> {predictions[0][0]} (Most Probable Next Word)".format(ctx=ctx, predictions=predictions))

    trainingResult = input("\nDid Amanu give you the correct prediction? (yes/no): ")
    if trainingResult.lower() == 'yes':
        with open(r'C:\Users\PYTHON PROGRAMS\AMANU\training.txt', 'a', encoding='utf-8') as file:
            file.write(f"{alphaSyllable} {predictions[0][0]}\n")
        print("Great! Amanu is learning and improving with your feedback.")
    else:
        with open(r'C:\Users\PYTHON PROGRAMS\AMANU\training.txt', 'a', encoding='utf-8') as file:
            file.write(f"{alphaSyllable} {predictions[1][0]}\n")
            file.write(f"{alphaSyllable} {predictions[2][0]}\n")
        print("Thank you for your feedback! Amanu will use this information to improve its predictions in the future.")

    print("\nYou can try entering different syllables or partial words to see how Amanu predicts the next word based on the context you provide. Enjoy exploring the Kapampangan language with Amanu!\n")

  
