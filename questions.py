from preprocessing import *

[train_unigram_without_PP, train_unigram_with_PP] = preprocess("brown-train.txt", "brown-train-PP.txt")
[b_test_unigram_without_PP, b_test_unigram_with_PP] = preprocess("brown-test.txt", "brown-test-PP.txt")
[l_test_unigram_without_PP, l_test_unigram_with_PP] = preprocess("learner-test.txt", "learner-test-PP.txt")


brown_train_types = len(train_unigram_with_PP)  # brown_train_types = len(b_train_original_dict) if we want the originals
sum_training_tokens = sum(train_unigram_with_PP.values())

print("Number of word types in training corpus, including padding symbols and the unknown token: "
      + str(len(train_unigram_with_PP)))
print("Total number of word tokens in training corpus: " + str(sum(train_unigram_with_PP.values())))

# Prior to mapping unknown words to <unk> in training and test data
unseens_in_brown = find_percent_of_unseens(train_unigram_without_PP, b_test_unigram_without_PP)
unseens_in_learners = find_percent_of_unseens(train_unigram_without_PP, l_test_unigram_without_PP)

print("Percentage of word tokens in Brown test corpus that did not occur in training: ", unseens_in_brown[1], '%')
print("Percentage of word tokens in Learner test corpus that did not occur in training: ", unseens_in_learners[1], '%')
print("Percentage of word types in Brown test corpus that did not occur in training: ", unseens_in_brown[0], '%')
print("Percentage of word types in Learner test corpus that did not occur in training: ", unseens_in_learners[0], '%')
