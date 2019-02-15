# Language Modeling

In this assignment, you will train several language models and will evaluate them on two test corpora. You can discuss in groups, but the homework is to be completed and submitted _individually_. Three files are provided with this assignment:

1. _brown-train.txt_
2. _brown-test.txt_
3. _learner-test.txt_

Each file is a collection of texts, one sentence per line. _Brown-train.txt_ contains 26,000 sentences from the [Brown corpus](http://clu.uni.no/icame/brown/bcm.html). You will use this corpus to train the language models. The test corpora ( _brown-test.txt_ and _learner-test.txt_ ) will be used to evaluate the language models that you trained. _brown-test.txt_ is a collection of sentences from the Brown corpus, different from the training data, and _learner-test.txt_ are essays written by non-native writers of English that are part of the [FCE corpus.](http://ilexir.co.uk/applications/clc-fce-dataset/)

## Preprocessing

Prior to training, please complete the following pre-processing steps:

1. Pad each sentence in the training and test corpora with start and end symbols (you can use \<s> and \<\/s>, respectively).
2. Lowercase all words in the training and test corpora. Note that the data already has been tokenized (i.e. the punctuation has been split off words).
3. Replace all words occurring in the training data once with the token \<unk>. Every word in the test data not seen in training should be treated as \<unk>.

## Training Models

Please use brown-train.txt to train the following language models:
1. A unigram maximum likelihood model.
2. A bigram maximum likelihood model.
3. A bigram model with Add-One smoothing.

## Questions

Please answer the questions below:

1. ( **5 points** ) How many word types (unique words) are there in the training corpus? Please include the padding symbols and the unknown token.
2. ( **5 points** ) How many word tokens are there in the training corpus?
3. ( **10 points** ) What percentage of word tokens and word types in each of the test corpora did not occur in training (before you mapped the unknown words to <unk> in training and test data)?
4. ( **20 points** ) What percentage of bigrams (bigram types and bigram tokens) in each of the test corpora that did not occur in training (treat <unk> as a token that has been observed).
5. ( **20 points** ) Compute the log probabilities of the following sentences under the three models (ignore capitalization and pad each sentence as described above). 
Please list all of the parameters required to compute the probabilities and show the complete calculation.
Which of the parameters have zero values under each model? Use log base 2 in your
calculations. Map words not observed in the training corpus to the <unk> token.
- He was laughed off the screen.
- There was no compulsion behind them.
- I look forward to hearing your reply.
6. ( **20 points** ) Compute the perplexities of each of the sentences above under each of the models.
7. ( **20 points** ) Compute the perplexities of the entire test corpora, separately for the _brown-test.txt_
and _learner-test.txt_ under each of the models. Discuss the differences in the results you
obtained.


## Submission

Please place the following on the **server venus.cs.qc.edu** and email me the path to the directory (in addition, please include all the required files in a tarball and email those to arozovskaya@qc.cuny.edu using subject lineCSCI381/CSCI780 Homework 1:

1. The Python code along with a README file that has instructions on how to run it in order to obtain the answers to questions in Section 1.
2. The writeup that includes the answers to the questions in Section 1.

Your grade will be based on the _correctness_ of your answers, the _clarity_ and completeness of your responses, and the _quality_ of the code that you submitted.

Please refer to the course webpage on late submission policy.

**Important:** Please make sure that you have uploaded the required homework files on the server. Your assignment will not be graded if the homework solution files are not accessible on the server.