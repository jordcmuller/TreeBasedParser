# the neural model for labelling the spans

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
import random

# constants
DIM_EMBEDDING = 100
EPOCHS = 100
GLOVE = "../data/glove.6B.100d.txt"

# Read Data
# train =
# dev =


# Set Up Indices

# this will correspond to the words from the sentences
id_to_token = [] # put in the appropriate tokens here
token_to_id = {} # set up the associated token to id index (must correspond with the id_to_token)

# this will correspond to the span labels
id_to_tag = []
tag_to_id = {}

NWORDS = len(id_to_token)
NTAGS = len(id_to_tag)

# for tokens, tags in train

# get GloVe embeddings
pretrained= {}
for line in open(GLOVE):
    parts = line.strip().split()
    word = parts[0]
    vector = [float(v) for v in parts[1:]]
    pretrained[word] = vector

pretrained_list = []


# data input
sentence = "input from the data"


# create set of all possible spans
def create_all_spans(sentence):
    all_spans = []

    for i in range(len(sentence)):
        for j in range(i+1, len(sentence)):
            all_spans.append([i, j])

    return all_spans

# IMPORTANT
# do we need to create the span-label pairs based on the gold output tree and the actual spans that it has?


# create set of spans and span labels from the set of all possible spans and the labels from the gold tree
# (input span, gold output label)

# we now have the list of span-label pairs for a sentence (whether this was from the data or was generated appropriately)
span_label_list = [["span1", "label1"], ["span2", "label2"]]

# put the span into the model word by word and then generate the labels and compare to the gold output label from
# the span_label_list

#
#
# MODEL
model = keras.Sequential()
# Layers:

#   Input — sequential
# Define the inputs
#
#   Embedding — What dimension input vectors
#             — embeddings will consist of a word vector and a PoS embedding concatenated (add this in later)
model.add(layers.Embedding(input_dim=DIM_EMBEDDING, output_dim=DIM_EMBEDDING))

# include the character level embeddings
# probably use a cnn for this


#   BLSTM — bidirectional LSTM
#         — How many hidden layers and units
#         — Only output after the whole span has been parsed, not at every input
model.add(
    layers.Bidirectional(layers.LSTM(64), input_dim=DIM_EMBEDDING)
)

#   Dense — What does this do?
#         — Will this actually just be the output layer?
#         — Number of units
model.add(layers.Dense(NTAGS))

#   Output — A unit for every type of output label
#
# Define/choose loss function
# Define/choose gradient method
#
#
# TRAINING FUNCTION #