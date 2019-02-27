from preprocessing import *
from modelling import *
#
# [test_without_PP, test_with_PP] = preprocess("test.txt", "testPP.txt")
[train_without_PP, train_with_PP, text] = preprocess("training.txt", "trainingPP.txt", None)
#
#
# print(train_with_PP)
# print(test_with_PP)
#
# print("Number of word types in training corpus: " + str(len(train_with_PP)))
# print("Total number word tokens in training corpus: " + str(sum(train_with_PP.values())))
#
# [unseen_types_percent, unseen_tokens_percent] = find_percent_of_unseens(train_without_PP, test_without_PP)
#
# print('unseen types percent', unseen_types_percent)
#
# print('unseen tokens percent', unseen_tokens_percent)

sentences = ["He was laughed off the screen . ",
             "There was no compulsion behind them . ",
             "I look forward to hearing your reply . "]
#
# for sentence in sentences:
#     print('For the sentence "' + sentence + '": ')
#     padded_text = pad_and_lowercase([sentence])
#     compute_unigram_log_prob(padded_text, train_with_PP)
#

unigram = unigram_model(train_with_PP)
padded_text = pad_and_lowercase([sentences[0]])
print(padded_text)
compute_unigram_log_prob(padded_text, unigram)
