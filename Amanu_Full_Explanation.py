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

BOS = "<s>" #Beginning of Sentence
EOS = "</s>" #End of Sentence

def tokenize(text: str) -> List[str]:
    # Tokenize the input text into words and punctuation
    return re.findall(r"[A-Za-z0-9']+|[.,!?;:]", text.lower())
    #this basically turns a text: "Hello, world!" into a List of tokens: ["hello", ",", "world", "!"]
    #pakamalan da ka, ya pin 

def make_ngrams(tokens: List[str], n: int) -> List[Tuple[str, ...]]:
    # Create n-grams from the list of tokens
    return [tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1)]
    #this takes a list of tokens and creates n-grams. For example, if n=2 (bigrams) and the tokens are ["hello", "world"], it will create [("hello", "world")]
    #in a unigram, it would create [("hello",), ("world",)], which will be used for a Kapampangan Language Predictor that uses an alphasyllabary input

class KapampanganLanguageModel:
    def __init__(self, n: int = 1, alpha: float = 0.5):
        self.n = n
        self.alpha = alpha
        self.vocab = set() #this creates a set to hold the unique words in the corpus. A set is used because it automatically handles duplicates, ensuring that each word is only stored once.
        self.ngram_counts: Dict[int, Counter] = defaultdict(Counter) 
        """
        this line initializes a dictionary that will hold the counts of n-grams. 
        The keys are the n-gram lengths (1 for unigrams, 2 for bigrams, etc.) and the values are Counter objects that count the occurrences of each n-gram.
        
        for example, if we have the sentence "Hello world", the bigram counts would be {2: Counter({("hello", "world"): 1})},
        and the unigram counts would be {1: Counter({("hello",): 1, ("world",): 1})}
        """
    def fit(self, corpus: List[str]) -> None:
        for line in corpus:
            toks = [BOS] * (self.n-1) + tokenize(line) + [EOS]
            """
            this line processes each line of the corpus by tokenizing it and adding special tokens for the beginning and end of the sentence. 
            The number of BOS tokens added depends on the value of n (the n-gram length). For example, if n=2 (bigrams), it will add one BOS token at the beginning. 
            
            If n=3 (trigrams), it will add two BOS tokens.
            For illustration, if the line is "Hello world" and n=2, toks will be ["<s>", "hello", "world", "</s>"]. If n=3, toks will be ["<s>", "<s>", "hello", "world", "</s>"]. 
            And in a unigram, it will just be ["hello", "world"] without any BOS tokens.
            """
            self.vocab.update(toks) 
            #this line updates the vocabulary set with the tokens from the current line. It ensures that all unique words from the corpus are added to the vocabulary.
            #for example, if the line is "Hello world", the vocabulary will be updated to include "hello" and "world". If the line is "Hello again", the vocabulary will be updated to include "again" as well.
            for k in range(1, self.n+1): 
                for ng in make_ngrams(toks, k):
                    self.ngram_counts[k][ng] += 1
                    """
                    #this nested loop creates n-grams for each line and updates the counts in the ngram_counts dictionary.
                    
                    #for example, if the line is "Hello world" and n=2, it will create the bigram ("hello", "world") and update its count. 
                    #If the line is "Hello again", it will create the bigram ("hello", "again") and update its count as well.
                    #the purpose of this is to build a frequency dictionary of n-grams that can be used later for making predictions based on the input sentence.
                    """
        self.Vocabulary = len(self.vocab) #this line calculates the size of the vocabulary (the number of unique words) and stores it in self.V. 
        #This is important for smoothing techniques when calculating probabilities, as it helps to account for unseen words in the corpus.

    def _conditional_logprob(self, word: str, history: Tuple[str, ...]) -> float:
        k = len(history) + 1 #history is the context of the word we want to predict. The length of the history determines the n-gram level we are working with (e.g., unigram, bigram, trigram).
        ngram = history + (word,) #this line creates the n-gram tuple by combining the history with the word we want to predict. For example, if the history is ("hello",) and the word is "world", the n-gram will be ("hello", "world").
        num = self.ngram_counts[k][ngram] + self.alpha
        """ 
        #this line calculates the numerator for the conditional probability of a word given its history. 
        #It retrieves the count of the n-gram (history + word) from the ngram_counts dictionary and applies add-alpha smoothing to handle cases where the n-gram may not have been seen in the training data.
        
        #for example, if we are calculating the probability of the word "world" given the history ("hello",) in a bigram model, it will look up the count of the bigram ("hello", "world") and add alpha to it.
        #if the bigram ("hello", "world") appears 5 times in the corpus and alpha is 0.5, the numerator would be 5 + 0.5 = 5.5.
        
        #in simple terms, this is counting how many times the word "world" follows "hello" in the corpus, and if it never does, it will still give a small probability due to the smoothing.
        """
        if k == 1:
            totalUnigrams = sum(self.ngram_counts[1].values())
            den = totalUnigrams + self.alpha * self.Vocabulary
            """
            #this block handles the case for unigrams (when k=1). It calculates the total count of all unigrams in the corpus and applies add-alpha smoothing to calculate the denominator for the probability.
            
            #for example, if we are calculating the probability of a unigram like "hello", it will sum up the counts of all unigrams and add alpha times the vocabulary size to account for unseen words.
            #if the "hello" unigram appears 10 times in the corpus, and there are 100 total unigrams, and the vocabulary size is 50 with alpha=0.5, the denominator would be 100 + (0.5 * 50) = 125.
            #then the probability of "hello" would be (10 + 0.5) / 125 which is equal to 0.084.
            """
        else:
            histCount = self.ngram_counts[k-1][history]
            #this line retrieves the count of the history (the context) from the ngram_counts dictionary for the (k-1)-gram level.
            #for example, if we are calculating the probability of "world" given the history ("hello",) in a bigram model, it will look up the count of the unigram ("hello",) to use as the denominator for the conditional probability.
            #if the history ("hello",) appears 20 times in the corpus, then histCount will be 20.
            den = histCount + self.alpha * self.Vocabulary
            """
            #this line calculates the denominator for the conditional probability of a word given its history for n-grams where k > 1.
            #it takes the count of the history (context) and applies add-alpha smoothing by adding  alpha times the vocabulary size.

            #for example, if the history ("hello",) appears 20 times in the corpus, and the vocabulary size is 50 with alpha=0.5, the denominator would be 20 + (0.5 * 50) = 45.
            #then, if the bigram ("hello", "world") appears 5 times, the probability of "world" given "hello" would be (5 + 0.5) / 45 which is equal to 0.122.

            #the difference with the unigram case is that here we are conditioning on a specific history, so we only consider the counts of that history rather than the total counts of all unigrams.
            """
        return math.log(num) - math.log(den)
        #this line calculates the log probability of the word given its history by taking the logarithm of the numerator and denominator.
        #using log probabilities is common in language modeling to avoid underflow issues when multiplying many small probabilities together.
        #for example, if the numerator is 5.5 and the denominator is 45, the log probability would be math.log(5.5) - math.log(45) which is approximately -1.204.

    def next_word_probability(self, context: List[str]) -> List[Tuple[str, float]]:
        hist = [BOS] * max(0, self.n-1 - len(context)) + context[-(self.n-1):]
        """
        #this line prepares the history (context) for making predictions. It adds the necessary number of BOS tokens at the beginning if the context is shorter than n-1, and then takes the last n-1 tokens from the context to form the history.
        #for example, if n=3 (trigrams) and the context is ["hello"], it will add two BOS tokens to make the history ["<s>", "<s>", "hello"]. If the context is ["hello", "world"], it will take the last two tokens to form the history ["hello", "world"].
        #if the context is empty, it will just be ["<s>", "<s>"] for a trigram model, which allows us to predict the first word of a sentence.
        #then, the model will use this history to calculate the probabilities of the next word based on the n-gram counts.
        """
        #SOFTMAX FUNCTION
        for hlen in range(self.n - 1, -1, -1):
            history = tuple(hist[-hlen:] if hlen > 0 else ())
            if hlen == 0 or (hlen in self.ngram_counts and self.ngram_counts[hlen].get(history, 0) > 0):
                #this line checks if the current history length (hlen) is valid for making predictions. It checks if hlen is 0 (which means we are using unigrams) or if the history exists in the n-gram counts for that length.
                #for example, if we are trying to predict the next word after "hello world" in a trigram model, it will first check for the history ("hello", "world"). If it exists, it will calculate the probabilities based on that. If not, it will check for the history ("world",) in a bigram model. If that also doesn't exist, it will fall back to unigram probabilities.
                #this allows the model to use the longest possible history for making predictions, which generally leads to better predictions, while still having a fallback to shorter histories if the longer ones are not available in the training data.
                logps = [(w, self._conditional_logprob(w, history)) for w in self.vocab]
                #the w is each word in the vocabulary, and self._conditional_logprob(w, history) calculates the log probability of that word given the current history.
                #this line creates a list of tuples where each tuple contains a word from the vocabulary and its corresponding log probability given the history. This will be used to rank the possible next words based on their probabilities.
                #for example, if the vocabulary includes "world", "again", and "friend", and the history is ("hello",), it will calculate the log probabilities for each of these words given the history ("hello",) and create a list like [("world", -1.204), ("again", -2.302), ("friend", -3.912)].
                
                #this line also calculates the log probabilities for all words in the vocabulary given the current history. It uses a list comprehension to iterate over each word in the vocabulary and calls the _conditional_logprob method to get the log probability of that word given the history.
                #for example, if the history is ("hello",) and the vocabulary includes "world", "again", and "friend", it will calculate the log probabilities for each of these words given the history ("hello",).
                m = max(lp for _, lp in logps)
                #this line finds the maximum log probability from the list of log probabilities. This is used for numerical stability when applying the softmax function, as it helps to prevent underflow issues when exponentiating small log probabilities.
                #for example, if the log probabilities are [("world", -1.204), ("again", -2.302), ("friend", -3.912)], the maximum log probability would be -1.204.
                exps = [(w, math.exp(lp - m)) for w, lp in logps]
                #this line applies the softmax function to convert the log probabilities into actual probabilities. It subtracts the maximum log probability from each log probability before exponentiating to improve numerical stability.
                #for example, if the log probabilities are [("world", -1.204), ("again", -2.302), ("friend", -3.912)], and the maximum log probability is -1.204, the exponentiated values would be [("world", 1.0), ("again", math.exp(-2.302 + 1.204)), ("friend", math.exp(-3.912 + 1.204))].
                Z = sum(v for _, v in exps)
                #this line calculates the normalization factor Z by summing up the exponentiated values. This ensures that the probabilities will sum to 1 after normalization.
                #for example, if the exponentiated values are [("world", 1.0), ("again", 0.1), ("friend", 0.01)], the normalization factor Z would be 1.0 + 0.1 + 0.01 = 1.11.
                return [(w, v / Z) for w, v in exps]
                #this line returns the list of words with their normalized probabilities by dividing each exponentiated value by the normalization factor Z.
                #for example, if the exponentiated values are [("world", 1. 0), ("again", 0.1), ("friend", 0.01)], and Z is 1.11, the returned probabilities would be [("world", 1.0 / 1.11), ("again", 0.1 / 1.11), ("friend", 0.01 / 1.11)] which are approximately [("world", 0.901), ("again", 0.090), ("friend", 0.009)].
            """
            #this loop block iterates over possible history lengths from n-1 down to 0 to find the longest matching history in the n-gram counts.
            #it checks if the history of length hlen exists in the n-gram counts. If it does, it calculates the log probabilities for all words in the vocabulary given that history, applies a softmax to convert them to probabilities, and returns the list of words with their probabilities.
            #if no matching history is found, it will eventually fall back to the unigram probabilities (when hlen=0), which gives a uniform distribution over the vocabulary.

            #for example, if we are trying to predict the next word after "hello world" in a trigram model, it will first check for the history ("hello", "world"). 
            #If it exists, it will calculate the probabilities based on that. If not, it will check for the history ("world",) in a bigram model. If that also doesn't exist, it will fall back to unigram probabilities.
            
            #the softmax step is used to convert log probabilities into actual probabilities that sum to 1, which allows us to rank the possible next words based on their likelihood given the context.
            #it is found at the end of the loop because we want to return the probabilities as soon as we find a valid history, starting from the longest possible history down to the shortest.
            #the formula used in the program is the standard softmax formula: P(w) = exp(logP(w) - m) / Z, where m is the maximum log probability to improve numerical stability, and Z is the normalization factor that ensures the probabilities sum to 1.

            """

            uni = 1.0 / self.Vocabulary 
            #this line calculates the uniform probability for unigrams, which is used as a fallback when no valid history is found in the n-gram counts. It assigns an equal probability to each word in the vocabulary.
            #for example, if the vocabulary size is 50, the uniform probability for each word would be 1.0 / 50 = 0.02. This means that if we have no information about the context, we would assign a 2% chance to each word in the vocabulary as the next word.
            return [(w, uni) for w in self.vocab]
        
    def predict_next(self, context: List[str], top_k: int = 5, exclude_punct: bool = True) -> List[Tuple[str, float]]:
        probs = self.next_word_probability(context)
        if exclude_punct:
            probs = [(w, p) for w, p in probs if w not in {".", ",", "!", "?", ";", ":"}]
            #this line filters out punctuation from the list of predicted next words if the exclude_punct flag is set to True. It creates a new list of tuples that only includes words that are not punctuation marks.
            #for example, if the predicted probabilities include [("world", 0.901), (",", 0.05), ("again", 0.045), ("!", 0.005)], and exclude_punct is True, the filtered list would be [("world", 0.901), ("again", 0.045)].
            #this is useful for language modeling tasks where we are primarily interested in predicting actual words rather than punctuation.:
        probs.sort(key=lambda x: x[1], reverse=True)
        #the reverse is set to True to sort the list in descending order, meaning that the words with the highest probabilities will appear at the top of the list.
        #this line sorts the list of predicted next words based on their probabilities in descending order. It uses a lambda function as the key to sort by the second element of each tuple (the probability).
        #for example, if the predicted probabilities are [("world", 0.901), ("again", 0.045)], after sorting, the list would remain [("world", 0.901), ("again", 0.045)] since "world" has a higher probability than "again".
        return probs[:top_k]
    
if __name__ == "__main__":
    with open(r'C:\Users\PYTHON PROGRAMS\training.txt', 'r', encoding='utf-8') as f:
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
        print("Great! Amanu is learning and improving with your feedback.")
    else:
        with open(r'C:\Users\PYTHON PROGRAMS\training.txt', 'a', encoding='utf-8') as file:
            file.write(f"{alphaSyllable} {predictions[1][0]}\n")
            file.write(f"{alphaSyllable} {predictions[2][0]}\n")
            #adds more predictions from the top 2 and top 3 to the training data for better learning
        print("Thank you for your feedback! Amanu will use this information to improve its predictions in the future.")

    print("\nYou can try entering different syllables or partial words to see how Amanu predicts the next word based on the context you provide. Enjoy exploring the Kapampangan language with Amanu!\n")

    """
    Tracing all steps from input to output:
    1. The user is prompted to enter a syllable or partial word.
    2. The input is tokenized using the tokenize function, which converts the input string into a list of tokens.
    3. The predict_next method of the KapampanganLanguageModel is called with the tokenized input as the context.
    4. Inside predict_next, the next_word_probability method is called to calculate the probabilities of the next words based on the context.
    5. The next_word_probability method prepares the history based on the context and iterates  over possible history lengths to find the longest matching history in the n-gram counts.
    6. If a valid history is found, it calculates the log probabilities for all words in the vocabulary given that history, applies a softmax to convert them to probabilities, and returns the list of words with their probabilities.
    7. If no valid history is found, it falls back to uniform probabilities for unigrams.
    8. The predicted next words are sorted by probability and the top predictions are returned. 
    """
