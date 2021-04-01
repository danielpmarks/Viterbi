# mp4.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created Fall 2018: Margaret Fleck, Renxuan Wang, Tiantian Fang, Edward Huang (adapted from a U. Penn assignment)
# Modified Spring 2020: Jialu Li, Guannan Guo, and Kiran Ramnath
# Modified Fall 2020: Amnon Attali, Jatin Arora
# Modified Spring 2021 by Kiran Ramnath
"""
Part 1: Simple baseline that only uses word statistics to predict tags
"""


def baseline(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences, each sentence is a list of (word,tag) pairs.
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''
    words = {}
    tags = {}
    for sentence in train:
        for word in sentence:
            if word[0] not in words:
                words[word[0]] = {}
                words[word[0]][word[1]] = 1
            else:
                # tag has not been seen for this word
                if word[1] not in words[word[0]]:
                    words[word[0]][word[1]] = 1
                # this word already has this tag -> add 1
                else:
                    words[word[0]][word[1]] = words[word[0]][word[1]] + 1
            if word[1] not in tags:
                tags[word[1]] = 1
            else:
                tags[word[1]] += 1

    common_tag = ""
    common_tag_count = 0
    for tag in tags:
        if tags[tag] > common_tag_count:
            common_tag = tag
            common_tag_count = tags[tag]

    train_tags = []
    for sentence in test:
        new_sentence = []
        for word in sentence:
            if word in words:
                max_tag = ""
                tag_count = 0
                for tag in words[word]:
                    if words[word][tag] > tag_count:
                        max_tag = tag
                        tag_count = words[word][tag]
                new_sentence.append((word, max_tag))
            else:
                new_sentence.append((word, common_tag))
        train_tags.append(new_sentence)

    return train_tags
