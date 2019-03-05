# Language Modeling

This project trains several language models and evaluates them on two test corpora. 

## How to Run

1. Verify that ``preprocessing.py``, ``modelling.py``, ``questions.py`` and the three corpora (``brown-train.txt``, ``brown-test.txt``, ``learner-test.txt``) are all in your current directory.
2. Run ``python3 questions.py`` in your terminal from the directory
3. The program will output four files:
    * ``output.txt`` has the answers to the questions below
    * ``brown-train-PP.txt`` has the preprocessed Brown training corpus.
    * ``brown-test-PP.txt`` has the preprocessed Brown test corpus.
    * ``learner-test-PP.txt`` has the preprocessed Learner test corpus.
4. Open ``output.txt`` to obtain the answers to the questions listed below

## Corpora
Each corpus is a collection of texts, one sentence per line. _Brown-train.txt_ contains 26,000 sentences from the Brown corpus. This corpus was used to train the language models. The test corpora ( _brown-test.txt_ and _learner-test.txt_ ) were used to evaluate the language models that were trained. _brown-test.txt_ is a collection of sentences from the Brown corpus, different from the training data, and _learner-test.txt_ are essays written by non-native writers of English that are part of the FCE Corpus.


## Preprocessing

Prior to training, the following pre-processing steps were completed:

1. Padding each sentence in the training and test corpora with start and end symbols (using \<s> and \<\/s>, respectively).
2. Lowercasing all words in the training and test corpora. Note that the data already has been tokenized (i.e. the punctuation has been split off words).
3. Replacing all words occurring in the training data once with the token \<unk>. Every word in the test data not seen in training was treated as \<unk>.

## Training Models

_brown-train.txt_ was used to train the following language models:
1. A unigram maximum likelihood model.
2. A bigram maximum likelihood model.
3. A bigram model with Add-One smoothing.

## Questions

The questions below were answered :

1. How many word types (unique words) are there in the training corpus? Please include the padding symbols and the unknown token.
2. How many word tokens are there in the training corpus?
3. What percentage of word tokens and word types in each of the test corpora did not occur in training (before you mapped the unknown words to \<unk> in training and test data)?
4. What percentage of bigrams (bigram types and bigram tokens) in each of the test corpora did not occur in training (treat \<unk> as a token that has been observed).
5. Compute the log probabilities of the following sentences under the three models (ignore capitalization and pad each sentence as described above). 
Please list all of the parameters required to compute the probabilities and show the complete calculation.
Which of the parameters have zero values under each model? Use log base 2 in your
calculations. Map words not observed in the training corpus to the <unk> token.
    - He was laughed off the screen .
    - There was no compulsion behind them .
    - I look forward to hearing your reply .
6. Compute the perplexities of each of the sentences above under each of the models.
7. Compute the perplexities of the entire test corpora, separately for the _brown-test.txt_
and _learner-test.txt_ under each of the models. Discuss the differences in the results you
obtained.