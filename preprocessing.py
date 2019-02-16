def preprocess(file_to_read, file_to_write):
    read = open(file_to_read, "r")
    new_text = ""

    for line in read.readlines():
        new_text += ("<s> " + line[:-1].lower() + " </s>\n")
    words_list = new_text.split()

    dict = {}

    # count up frequencies for words in text
    for word in words_list:
        if word in dict:
            dict[word] += 1
        else:
            dict[word] = 1

    # replace words which occur just once with <unk> token
    for i in range(len(words_list)):
        if dict[words_list[i]] == 1:
            words_list[i] = "<unk>"

    # count up frequencies of words in the text again, this time with <unk>

    replaced_dict = {}

    for word in words_list:
        if word in replaced_dict:
            replaced_dict[word] += 1
        else:
            replaced_dict[word] = 1

    write_file = open(file_to_write, "w")

    for word in words_list:
        write_file.write(word + ' ')
    write_file.close()


preprocess("brown-train.txt", "brown-train-PP.txt")
preprocess("brown-test.txt", "brown-test-PP.txt")
preprocess("learner-test.txt", "learner-test-PP.txt")