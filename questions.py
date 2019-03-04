from preprocessing import *
from modelling import *

# Preprocessing
[train_frequency_before_unk, train_frequency, train_text] = \
    preprocess("brown-train.txt", "brown-train-PP.txt", None)

[b_test_frequency_before_unk, b_test_frequency, brown_test_text] = \
    preprocess("brown-test.txt", "brown-test-PP.txt", train_frequency)

[l_test_frequency_before_unk, l_test_frequency, learner_test_text] = \
    preprocess("learner-test.txt", "learner-test-PP.txt", train_frequency)

# Modelling

unigram = unigram_model(train_frequency)
bigram_MLE = bigram_model(train_frequency, train_text, False)
bigram_add1 = bigram_model(train_frequency, train_text, True)

# 1
print("Question 1: Number of word types in training corpus, including padding symbols and the unknown token:",
      len(train_frequency))

# 2
print("Question 2: Total number of word tokens in training corpus:", (sum(train_frequency.values())))

# 3   Prior to mapping unknown words to <unk> in training and test data
print("Question 3: ")
unseen_words_in_brown = find_percent_of_unseen_words(train_frequency_before_unk, b_test_frequency_before_unk)
unseen_words_in_learners = find_percent_of_unseen_words(train_frequency_before_unk, l_test_frequency_before_unk)

print("Percentage of word types in Brown test corpus that did not occur in training: ",
      unseen_words_in_brown[0], '%')
print("Percentage of word tokens in Brown test corpus that did not occur in training: ",
      unseen_words_in_brown[1], '%')
print("Percentage of word types in Learner test corpus that did not occur in training: ",
      unseen_words_in_learners[0], '%')
print("Percentage of word tokens in Learner test corpus that did not occur in training: ",
      unseen_words_in_learners[1], '%')

# 4
print("Question 4: ")
unseen_bigrams_in_brown = find_percent_of_unseen_bigrams(bigram_MLE, brown_test_text)
unseen_bigrams_in_learners = find_percent_of_unseen_bigrams(bigram_MLE, learner_test_text)

print("Percentage of bigram types in Brown test corpus that did not occur in training: ",
      unseen_bigrams_in_brown[0], '%')
print("Percentage of bigram tokens in Brown test corpus that did not occur in training: ",
      unseen_bigrams_in_brown[1], '%')
print("Percentage of bigram types in Learner test corpus that did not occur in training: ",
      unseen_bigrams_in_learners[0], '%')
print("Percentage of bigram tokens in Learner test corpus that did not occur in training: ",
      unseen_bigrams_in_learners[1], '%')

# print("Question 5A: ")
# # 5   Compute the log probabilities of the following sentences under the three models
# #     (ignore capitalization and pad each sentence as described above).
# #     Please list all of the parameters required to compute the probabilities and show the complete calculation.
# #     Which of the parameters have zero values under each model? Use log base 2 in your calculations.
# #     Map words not observed in the training corpus to the <unk> token.
#
# sentences = ["He was laughed off the screen . ",
#              "There was no compulsion behind them . ",
#              "I look forward to hearing your reply . "]
#
# for sentence in sentences:
#     print('For the sentence "' + sentence + '": ')
#     padded_text = pad_and_lowercase([sentence])
#     compute_unigram_log_prob(padded_text, train_frequency)
#
# # compute_unigram_log_prob([learner_test_text], train_unigram_with_PP)