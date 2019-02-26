def pad_and_lowercase(lines):  # returns a list of padded and lowercased words
    new_text = ""
    for line in lines:
        new_text += (" <s> " + line.lower() + " </s> ")
    return new_text.split()


def count_frequencies(text):
    freq = {}
    for word in text:
        if word in freq:
            freq[word] += 1
        else:
            freq[word] = 1
    return freq


def preprocess(file_to_read, file_to_write, training_corpus):
    read = open(file_to_read, "r")
    words = pad_and_lowercase(read.readlines())
    original_freq = count_frequencies(words)

    if training_corpus is None:  # if this is a training corpus
        # replace words which occur just once with <unk> token
        for i in range(len(words)):
            if original_freq[words[i]] == 1:
                words[i] = "<unk>"

    else:  # if this is a test corpus
        # replace words which occur just once with <unk> token
        for i in range(len(words)):
            if words[i] not in training_corpus:
                words[i] = "<unk>"

    # count up frequencies of words in the text again, this time with <unk>
    replaced_freq = count_frequencies(words)

    write_file = open(file_to_write, "w")
    text = ' '.join(words)
    write_file.write(text)
    write_file.close()

    return [original_freq, replaced_freq, text]
