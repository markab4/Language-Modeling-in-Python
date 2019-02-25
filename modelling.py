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


def compute_unigram_log_prob(text, corpus):
    freq = {'<unk>': 0}
    for word in text:
        if word in corpus:
            if word in freq:
                freq[word] += 1
            else:
                freq[word] = 1
        else:
            freq['<unk>'] += 1
    n = sum(corpus.values())
    print('The frequencies of each parameter needed to compute the probabilities are:', freq)
    print('The total # of tokens is:', n)

    prob_freq = {word: count/n for word, count in freq.items()}
    print('The probabilities (frequency/total # of tokens) are:', prob_freq)

    log_freq = {word: math.log(count, 2) for word, count in prob_freq.items()}
    print('The log (base 2) probabilities are:', log_freq)

    log_prob = sum(log_freq.values())
    print('The sum of these log probabilities (and the log probability of this sentence) is: ', log_prob)
    return freq