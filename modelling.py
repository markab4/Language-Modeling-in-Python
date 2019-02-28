from preprocessing import *
import math


# What percentage of word tokens and word types in each of the test corpora did not occur in training
# (before mapping the unknown words to <unk> in training and test data)
def find_percent_of_unseens(training_dict, test_dict):
    sum_of_unseen_tokens = 0
    number_of_unseen_types = 0
    number_of_tokens_in_test = sum(test_dict.values())
    number_of_types_in_test = len(test_dict)

    for word in test_dict:
        if word not in training_dict:
            sum_of_unseen_tokens += test_dict[word]
            number_of_unseen_types += 1

    unseen_types_percent = number_of_unseen_types / number_of_types_in_test * 100
    unseen_tokens_percent = sum_of_unseen_tokens / number_of_tokens_in_test * 100
    return [unseen_types_percent, unseen_tokens_percent]


def unigram_model(training_dictionary):
    unigram = dict(training_dictionary)
    unigram.pop('<s>', None)
    n = sum(unigram.values())
    return {word: math.log(count/n, 2) for word, count in unigram.items()}


def bigram_model(frequency, text):  # WordFrequency[token[i]][token[i+1]] += 1
    bigram = {}
    for i in range(len(text[:-1])):         # goes from index 0 to second to last element
        if text[i] not in bigram:           # if bigram has doesnt have the key of the first word
            bigram[text[i]] = {}                    # initialize the dictionary
        if text[i+1] in bigram[text[i]]:    # if word i has key word i + 1
            bigram[text[i]][text[i+1]] += 1 / frequency[text[i]]
        else:
            bigram[text[i]][text[i + 1]] = 1 / frequency[text[i]]
    return bigram


def compute_unigram_log_prob(text, unigram):
    subset = {word if word in unigram else '<unk>': unigram[word if word in unigram else '<unk>'] for word in text}
    print('The parameters required to compute the probabilities are: ', subset)

    log_prob = sum(subset.values())
    print('The sum of these log probabilities (and the log probability of this text) is: ', log_prob)

    m = len(text)
    avg_log_prob = log_prob/m
    print('The average log probability for this text is: ', avg_log_prob)

    perplexity = 2 ** (-avg_log_prob)
    print('The perplexity for this text is: ', perplexity)
