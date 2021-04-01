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
This file should not be submitted - it is only meant to test your implementation of the Viterbi algorithm. 

See Piazza post @650 - This example is intended to show you that even though P("back" | RB) > P("back" | VB), 
the Viterbi algorithm correctly assigns the tag as VB in this context based on the entire sequence. 
"""
from utils import read_files, get_nested_dictionaries
import math


class trellis_cell:
    def __init__(self, tag, val, ptr):
        self.tag = tag
        self.val = val
        self.ptr = ptr


def main():
    test, emission, transition, output = read_files()
    emission, transition = get_nested_dictionaries(emission, transition)
    initial = transition["START"]
    prediction = []
    print(emission)

    trellis = {}

    # initialize the trellis
    for tag in initial:
        trellis[tag] = trellis_cell(
            tag, initial[tag]*emission[tag][test[0][0]], None)

    last_cell = None
    i = 1
    while i < len(test[0]):

        # Initialize next layer
        next_layer = {}
        # Loop through layer
        for next_tag in trellis:
            max_prob = 0
            prev_cell = None
            # Loop through the current node layer
            for tag in trellis:
                # Calculate the edge probability
                prob = trellis[tag].val*transition[tag][next_tag] * \
                    emission[next_tag][test[0][i]]
                # Keep track of max probability
                if prob > max_prob:
                    max_prob = prob
                    prev_cell = trellis[tag]
            # Update node layer with the maximum probability
            next_layer[next_tag] = trellis_cell(next_tag, max_prob, prev_cell)

        # Find the max node value in the layer
        max_val = 0
        tag_choice = ""
        cell_choice = None
        for tag in next_layer:
            if next_layer[tag].val > max_val:
                max_val = next_layer[tag].val
                tag_choice = tag
                cell_choice = next_layer[tag]

        trellis = next_layer
        last_cell = cell_choice
        i += 1

    # Backtrace to find the optimal route
    i -= 1
    while last_cell != None:
        prediction.insert(0, (test[0][i], last_cell.tag))
        last_cell = last_cell.ptr
        i -= 1

    print('Your Output is:', prediction, '\n Expected Output is:', output)


if __name__ == "__main__":
    main()
