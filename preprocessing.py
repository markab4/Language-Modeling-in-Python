def preprocess(file_to_read, file_to_write):
    read = open(file_to_read, "r")
    new_text = ""

    lines = read.readlines()

    for line in lines:
        new_text += (" <s> " + line.lower() + " </s> ")

    words_list = new_text.split()

    original_dict = {}

    # count up frequencies for words in text
    for word in words_list:
        if word in original_dict:
            original_dict[word] += 1
        else:
            original_dict[word] = 1

    # replace words which occur just once with <unk> token
    for i in range(len(words_list)):
        if original_dict[words_list[i]] == 1:
            words_list[i] = "<unk>"

    # count up frequencies of words in the text again, this time with <unk>

    replaced_dict = {}

    for word in words_list:
        if word in replaced_dict:
            replaced_dict[word] += 1
        else:
            replaced_dict[word] = 1

    write_file = open(file_to_write, "w")

    write_file.write(' '.join(words_list))

    # for word in words_list:
    #     write_file.write(word + ' ')
    write_file.close()

    return [original_dict, replaced_dict]


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

