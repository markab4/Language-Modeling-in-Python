from preprocessing import *
import math


# returns dictionary of unigram model: (count of word in corpus)/(number of words in corpus)
def unigram_model(training_dictionary):
    unigram = dict(training_dictionary)
    unigram.pop('<s>', None)
    n = sum(unigram.values())
    return {word: count/n for word, count in unigram.items()}


# returns dictionary of dictionaries for both bigram MLE and add-one
def bigram_model(frequency, text, add_one):
    bigram = dict(frequency)
    vocabulary = len(frequency)
    bucket = {word: (1 if add_one else 0) for word, count in frequency.items()}
    for word in bigram:
        bigram[word] = dict(bucket)
    for i in range(len(text)-1):         # goes from index 0 to second to last element
        bigram[text[i]][text[i + 1]] += 1 / (frequency[text[i]] + (vocabulary if add_one else 0))
    # for bucket in bigram:
    #     bigram[bucket] = {word: math.log(count, 2) for word, count in bigram[bucket].items()}
    return bigram


# What percentage of word tokens and word types in each of the test corpora did not occur in training
# (before mapping the unknown words to <unk> in training and test data)
def find_percent_of_unseen_words(training_dict, test_dict):
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


#  What percentage of bigrams (bigram types and bigram tokens) in each of the test corpora
#  did not occur in training (treat <unk> as a token that has been observed).
def find_percent_of_unseen_bigrams(test_frequency, bigram, text):
    unseen_types = {}
    seen_types = {}
    sum_of_unseen_tokens = 0
    number_of_tokens_in_test = 0

    for i in range(len(text)-1):
        number_of_tokens_in_test += 1
        if bigram[text[i]][text[i+1]] == 0:
            sum_of_unseen_tokens += 1
            if text[i] not in unseen_types:      # if bigram has doesnt have the key of the first word
                unseen_types[text[i]] = set()
            unseen_types[text[i]].add(text[i+1])
        else:
            if text[i] not in seen_types:
                seen_types[text[i]] = set()
            seen_types[text[i]].add(text[i+1])

    unseen = {w1: len(w2) for w1, w2 in unseen_types.items()}
    number_of_unseen_types = sum(unseen.values())
    seen = {w1: len(w2) for w1, w2 in seen_types.items()}
    number_of_seen_types = sum(seen.values())
    number_of_types_in_test = number_of_unseen_types + number_of_seen_types

    unseen_types_percent = number_of_unseen_types / number_of_types_in_test * 100
    unseen_tokens_percent = sum_of_unseen_tokens / number_of_tokens_in_test * 100
    return [unseen_types_percent, unseen_tokens_percent]


def compute_unigram_log_prob(text, unigram):
    subset = {word if word in unigram else '<unk>': unigram[word if word in unigram else '<unk>'] for word in text}
    print('The parameters required to compute the probabilities are:', subset)

    log_prob = {word: math.log(count, 2) for word, count in unigram.items()}
    print('The log base 2 of each of these probabilities is:', log_prob)

    sum_log_prob = sum(subset.values())
    print('The sum of these log probabilities (and the log probability of this text) is:', sum_log_prob)

    m = len(text)
    avg_log_prob = sum_log_prob/m
    print('The average log probability for this text is: ', avg_log_prob)

    perplexity = 2 ** (-avg_log_prob)
    print('The perplexity for this text is: ', perplexity)
