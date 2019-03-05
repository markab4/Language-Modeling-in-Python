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


output = open("output.txt", "w")

# 1

output.write("Question 1: \n"
             "Number of word types in training corpus, including padding symbols and the unknown token: \t" +
             str(len(train_frequency)) + '\n')

# 2
output.write("\nQuestion 2: \n"
             "Total number of word tokens in training corpus: \t" +
             str(sum(train_frequency.values())) + '\n')

# 3   Prior to mapping unknown words to <unk> in training and test data
output.write("\nQuestion 3:\n")

unseen_words_in_brown = find_percent_of_unseen_words(train_frequency_before_unk, b_test_frequency_before_unk)
unseen_words_in_learners = find_percent_of_unseen_words(train_frequency_before_unk, l_test_frequency_before_unk)

output.write("Percentage of word types in Brown test corpus that did not occur in training:\t" +
             str(unseen_words_in_brown[0]) + '%\n')
output.write("Percentage of word tokens in Brown test corpus that did not occur in training:\t" +
             str(unseen_words_in_brown[1]) + '%\n')
output.write("Percentage of word types in Learner test corpus that did not occur in training:\t" +
             str(unseen_words_in_learners[0]) + '%\n')
output.write("Percentage of word tokens in Learner test corpus that did not occur in training:\t" +
             str(unseen_words_in_learners[1]) + '%\n')

# 4
output.write("\nQuestion 4:\n")
unseen_bigrams_in_brown = find_percent_of_unseen_bigrams(bigram_MLE, brown_test_text)
unseen_bigrams_in_learners = find_percent_of_unseen_bigrams(bigram_MLE, learner_test_text)

output.write("Percentage of bigram types in Brown test corpus that did not occur in training:\t" +
             str(unseen_bigrams_in_brown[0]) + '%\n')
output.write("Percentage of bigram tokens in Brown test corpus that did not occur in training:\t" +
             str(unseen_bigrams_in_brown[1]) + '%\n')
output.write("Percentage of bigram types in Learner test corpus that did not occur in training:\t" +
             str(unseen_bigrams_in_learners[0]) + '%\n')
output.write("Percentage of bigram tokens in Learner test corpus that did not occur in training:\t" +
             str(unseen_bigrams_in_learners[1]) + '%\n')

#5
output.write("\nQuestions 5 and 6:\n")
#     Compute the log probabilities of the following sentences under the three models
#     (ignore capitalization and pad each sentence as described above).

#     List all of the parameters required to compute the probabilities and show the complete calculation.

#     Which of the parameters have zero values under each model? Use log base 2 in your calculations.
#     Map words not observed in the training corpus to the <unk> token.

sentences = ["He was laughed off the screen . ",
             "There was no compulsion behind them . ",
             "I look forward to hearing your reply . "]

for sentence in sentences:
    output.write('\nFor the sentence "' + str(sentence) + '": \n')
    output.write(compute_unigram_log_prob(sentence, unigram))
    output.write(compute_bigram_MLE_log_prob(sentence, bigram_MLE))
    output.write(compute_bigram_add1_log_prob(sentence, bigram_add1))

output.close()
