from preprocessing import *
from modelling import *

[train_unigram_without_PP, train_unigram_with_PP, train_text] = preprocess("brown-train.txt", "brown-train-PP.txt", None)
[b_test_unigram_without_PP, b_test_unigram_with_PP, brown_test_text] = preprocess("brown-test.txt", "brown-test-PP.txt", train_unigram_with_PP)
[l_test_unigram_without_PP, l_test_unigram_with_PP, learner_test_text] = preprocess("learner-test.txt", "learner-test-PP.txt", train_unigram_with_PP)


# 1
print("Number of word types in training corpus, including padding symbols and the unknown token:",
      len(train_unigram_with_PP))

# 2
print("Total number of word tokens in training corpus:", (sum(train_unigram_with_PP.values())))

# 3   Prior to mapping unknown words to <unk> in training and test data
unseens_in_brown = find_percent_of_unseens(train_unigram_without_PP, b_test_unigram_without_PP)
unseens_in_learners = find_percent_of_unseens(train_unigram_without_PP, l_test_unigram_without_PP)

print("Percentage of word tokens in Brown test corpus that did not occur in training: ", unseens_in_brown[1], '%')
print("Percentage of word tokens in Learner test corpus that did not occur in training: ", unseens_in_learners[1], '%')
print("Percentage of word types in Brown test corpus that did not occur in training: ", unseens_in_brown[0], '%')
print("Percentage of word types in Learner test corpus that did not occur in training: ", unseens_in_learners[0], '%')

# 4   Compute the log probabilities of the following sentences under the three models
#     (ignore capitalization and pad each sentence as described above).
#     Please list all of the parameters required to compute the probabilities and show the complete calculation.
#     Which of the parameters have zero values under each model? Use log base 2 in your calculations.
#     Map words not observed in the training corpus to the <unk> token.

sentences = ["He was laughed off the screen . ",
             "There was no compulsion behind them . ",
             "I look forward to hearing your reply . "]

for sentence in sentences:
    print('For the sentence "' + sentence + '": ')
    padded_text = pad_and_lowercase([sentence])
    compute_unigram_log_prob(padded_text, train_unigram_with_PP)

# compute_unigram_log_prob([learner_test_text], train_unigram_with_PP)