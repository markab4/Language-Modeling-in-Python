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


def preprocess(file_to_read, file_to_write):
    read = open(file_to_read, "r")
    text = pad_and_lowercase(read.readlines())
    original_freq = count_frequencies(text)

    # replace words which occur just once with <unk> token
    for i in range(len(text)):
        if original_freq[text[i]] == 1:
            text[i] = "<unk>"

    # count up frequencies of words in the text again, this time with <unk>
    replaced_freq = count_frequencies(text)

    write_file = open(file_to_write, "w")
    write_file.write(' '.join(text))
    write_file.close()

    return [original_freq, replaced_freq]