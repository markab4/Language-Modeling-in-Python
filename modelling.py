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
        bigram[text[i]][text[i + 1]] += 1
    for bucket in bigram:
        bigram[bucket] = {word: count/(frequency[bucket] + (vocabulary if add_one else 0)) for word, count in bigram[bucket].items()}
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
def find_percent_of_unseen_bigrams(bigram, text):
    unseen_types = {}
    seen_types = {}
    sum_of_unseen_tokens = 0
    number_of_tokens_in_test = len(text) - 1

    for i in range(len(text)-1):
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


def compute_unigram_log_prob(sentence, unigram):
    computation = '\nIn the Unigram Model: \n'
    text = (sentence.lower() + " </s> ").split()
    replace_unseens_with_unk(text, unigram)
    computation += "The sentence gets mapped to " + str(text) + '\n'

    subset = {word: unigram[word] for word in text}
    computation += 'The parameters required to compute the probabilities are:\n' + pretty_dict(subset)

    log_prob = {word: math.log(count, 2) for word, count in subset.items()}
    computation += '\nThe log base 2 of each of these probabilities is:\n' + pretty_dict(log_prob)

    sum_log_prob = 0
    for word in text:
        sum_log_prob += log_prob[word]

    computation += '\nThe sum of the log probabilities (and the log probability of this text) is:\t' + \
                   str(sum_log_prob)

    m = len(text)
    avg_log_prob = sum_log_prob/m
    computation += '\nThe average log probability for this text is:\t' + str(avg_log_prob)

    perplexity = 2 ** (-avg_log_prob)
    computation += '\nThe perplexity for this text is:\t' + str(perplexity) + '\n'

    return computation


def compute_bigram_MLE_log_prob(sentence, bigram):
    computation = '\nIn the Bigram MLE Model: \n'
    text = pad_and_lowercase([sentence])
    replace_unseens_with_unk(text, bigram)
    computation += "The sentence gets mapped to " + str(text) + '\n'
    subset = set()
    sum_log_prob = 0
    has_zero = False
    for i in range(len(text)-1):
        subset.add((text[i], text[i+1], bigram[text[i]][text[i+1]]))
        if bigram[text[i]][text[i+1]] == 0:
            has_zero = True
        else:
            sum_log_prob += math.log(bigram[text[i]][text[i+1]], 2)

    if has_zero:
        computation += "The following parameters need to be computed:\n" + pretty_bigram(subset) + \
                       "\nThe log probability is undefined due to the following unseen bigrams:\n"
        for tup in subset:
            if tup[2] == 0:
                computation += '"' + str(tup[0]) + " " + str(tup[1]) + '"\n'

    else:
        computation += "The following parameters need to be computed:\n"
        log_prob = set()
        for bigram in subset:
            computation += '"' + str(bigram[0]) + " " + str(bigram[1]) + '" : ' + str(bigram[2]) + '\n'
            log_prob.add((bigram[0], bigram[1], math.log(bigram[2], 2)))

        computation += '\nThe log base 2 of each of these probabilities is:\n' + pretty_bigram(log_prob) + \
                       '\nThe sum of the log probabilities (and the log probability of this text) is:\t' + \
                       str(sum_log_prob)

        m = len(text)
        avg_log_prob = sum_log_prob / m
        computation += '\nThe average log probability for this text is:\t' + str(avg_log_prob)

        perplexity = 2 ** (-avg_log_prob)
        computation += '\nThe perplexity for this text is:\t' + str(perplexity) + '\n'
    return computation


def compute_bigram_add1_log_prob(sentence, bigram):
    computation = '\nIn the Bigram Add-One Model: \n'
    text = pad_and_lowercase([sentence])
    replace_unseens_with_unk(text, bigram)
    computation += "The sentence gets mapped to " + str(text) + '\n'
    subset = set()
    sum_log_prob = 0
    for i in range(len(text)-1):
        subset.add((text[i], text[i+1], bigram[text[i]][text[i+1]]))
        sum_log_prob += math.log(bigram[text[i]][text[i+1]], 2)

    computation += "The following parameters need to be computed:\n"
    log_prob = set()
    for bigram in subset:
        computation += '"' + str(bigram[0]) + " " + str(bigram[1]) + '" : ' + str(bigram[2]) + '\n'
        log_prob.add((bigram[0], bigram[1], math.log(bigram[2], 2)))

    computation += '\nThe log base 2 of each of these probabilities is:\n' + pretty_bigram(log_prob) + \
                   '\nThe sum of the log probabilities (and the log probability of this text) is:\t' + \
                   str(sum_log_prob)

    m = len(text)
    avg_log_prob = sum_log_prob/m
    computation += '\nThe average log probability for this text is:\t' + str(avg_log_prob)

    perplexity = 2 ** (-avg_log_prob)
    computation += '\nThe perplexity for this text is:\t' + str(perplexity) + '\n'

    return computation
