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
# Modified Spring 2021 by Kiran Ramnath (kiranr2@illinois.edu)

"""
Part 3: Here you should improve viterbi to use better laplace smoothing for unseen words
This should do better than baseline and your first implementation of viterbi, especially on unseen words
"""

import math


def viterbi_2(train, test):
    '''
    input:  training data (list of sentences, with tags on the words)
            test data (list of sentences, no tags on the words)
    output: list of sentences with tags on the words
            E.g., [[(word1, tag1), (word2, tag2)], [(word3, tag3), (word4, tag4)]]
    '''

    k = 1e-10
    word_counts, unique_words, unique_tags, tags, pair_counts, hapax = count_tags(
        train)
    num_words = len(unique_words)

    # Find the tag probability
    emission = {}
    for tag in word_counts:
        emission[tag] = {}
        for word in word_counts[tag]:
            emission[tag][word] = math.log((word_counts[tag][word] + k) /
                                           (tags[tag] + k * (len(word_counts[tag]) + 1)))

    # Find the tag pair probabilities P(Tn | Tn-1)
    transition = {}
    for tag in pair_counts:
        transition[tag] = {}
        for next_tag in tags:
            count = 0
            if next_tag in pair_counts[tag]:
                count = pair_counts[tag][next_tag]
            transition[tag][next_tag] = math.log((
                count + k) / (tags[tag] + k*(len(pair_counts[tag])+1)))

    predictions = viterbi(test, emission, transition,
                          tags, k, hapax, word_counts)
    # print(predictions[0])
    return predictions


def count_tags(train):
    '''
    Builds a trained model to input to the viterbi model
    inputs: the training set,
    '''

    word_counts_by_tag = {}
    unique_words = {}
    unique_tags = {}
    tag_counts = {}
    pair_counts = {}
    # overwritten for every tag, only useful for hapax
    tags = {}
    word_counts = {}
    for sentence in train:
        for i in range(len(sentence) - 1):
            word = sentence[i]
            next_word = sentence[i+1]

            # save tag for hapax
            tags[word[0]] = word[1]
            if word[0] not in word_counts:
                word_counts[word[0]] = 0
            word_counts[word[0]] += 1

            if word[0] not in unique_words:
                unique_words[word[0]] = 1
            if word[1] not in unique_tags:
                unique_tags[word[1]] = 1

            if word[1] in word_counts_by_tag:
                if word[0] not in word_counts_by_tag[word[1]]:
                    word_counts_by_tag[word[1]][word[0]] = 1
                else:
                    word_counts_by_tag[word[1]][word[0]] += 1
            else:
                word_counts_by_tag[word[1]] = {}
                word_counts_by_tag[word[1]][word[0]] = 1

            if word[1] not in tag_counts:
                tag_counts[word[1]] = 1
            else:
                tag_counts[word[1]] += 1

            if word[1] not in pair_counts:
                pair_counts[word[1]] = {}
                pair_counts[word[1]][next_word[1]] = 1
            else:
                if next_word[1] not in pair_counts[word[1]]:
                    pair_counts[word[1]][next_word[1]] = 1
                else:
                    pair_counts[word[1]][next_word[1]] += 1
    # print(pair_counts)
    hapax = {}
    for word in word_counts:
        # print(word)
        if word_counts[word] == 1:

            if tags[word] not in hapax:
                hapax[tags[word]] = 0
            hapax[tags[word]] += 1
    for tag in tag_counts:
        if tag not in hapax:
            hapax[tag] = 1
    # print(hapax)

    return word_counts_by_tag, unique_words, unique_tags, tag_counts, pair_counts, hapax

# Viterbi trellis node class


class trellis_cell:
    def __init__(self, tag, val, ptr):
        self.tag = tag
        self.val = val
        self.ptr = ptr


def viterbi(test, emission, transition, tags, k, hapax, word_counts):
    '''
        Calculates the viterbi output from a training set
        input: emission and trainsition model
        output: the prediction sequence from the model
    '''

    initial = transition['START']

    # print(initial)
    predictions = []
    sen_num = 0
    for sentence in test:
        # print("NEXT SENTENCE")
        sen_num += 1
        # if sen_num == 5:
        #    break
        cur_layer = {}

        # print("INITIAL")
        # initialize the trellis
        for tag in initial:
            if tag != 'START':
                p_first_word = 0
                if sentence[1] in emission[tag]:
                    p_first_word = emission[tag][sentence[1]]
                    # print("Found ", p_first_word)
                    # print(math.log((k) / (tags[tag] + k*(len(tags)+1))))
                else:
                    p_first_word = math.log(
                        (k * hapax[tag]) / (tags[tag] + k * hapax[tag] * (len(emission[tag]) + 1)))
                # print("         ", tag, " ", initial[tag]+p_first_word)
                cur_layer[tag] = trellis_cell(
                    tag, initial[tag]+p_first_word, None)

        last_cell = None
        i = 2
        # print(sentence)
        while i < len(sentence):

            # print("i: ", i, "Len: ", len(sentence))
            # print(sentence[i])
            # Initialize next layer
            next_layer = {}
            # Loop through layer
           # print("Looping through next layer")
            for next_tag in tags:
                max_prob = -99999999999999
                prev_cell = None
                # Loop through the current node layer
                for tag in cur_layer:
                    p_word = 0

                    if sentence[i] in emission[next_tag]:
                        p_word = emission[next_tag][sentence[i]]

                        # print("0")
                    else:
                        # print("unknown")
                        p_word = math.log(
                            (k * hapax[next_tag]) / (tags[next_tag] + k*hapax[next_tag]*(len(tags)+1)))
                    # print(tag, " ", p_word)
                    # Calculate the edge probability

                    prob = cur_layer[tag].val+transition[tag][next_tag] + \
                        p_word
                    # print(tag, " ", next_tag, " ", transition[tag][next_tag])
                    # print(tag, " ", p_word)
                    # Keep track of max probability
                    if prob > max_prob:
                        max_prob = prob
                        prev_cell = cur_layer[tag]
                # Update node layer with the maximum probability
                next_layer[next_tag] = trellis_cell(
                    next_tag, max_prob, prev_cell)
                # print(next_tag, " ", prev_cell.tag)

            # print(cell_choice.tag)
            cur_layer = next_layer
            # print(cell_choice.tag)
            i += 1

        max_val = -9999999999
        for tag in cur_layer:
            if cur_layer[tag].val > max_val:
                max_val = cur_layer[tag].val
                last_cell = cur_layer[tag]

        # Backtrace to find the optimal route
        last_cell.tag = 'END'
        i -= 1
        prediction = []
        while last_cell != None:
            # print(sentence[i], " ", last_cell.tag)
            prediction.insert(0, (sentence[i], last_cell.tag))
            last_cell = last_cell.ptr
            i -= 1
        prediction.insert(0, ('START', 'START'))
        # print()
        predictions.append(prediction)
        # print(prediction)

    return predictions
