# the neural model for labelling the spans
# parts of this code is adapted from https://jkk.name/neural-tagger-tutorial/

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
import numpy as np
import random

# constants
DIM_EMBEDDING = 100
EPOCHS = 100
GLOVE = "../Embeddings/glove.6B.100d.txt"

train_path = r"C:\Users\jorda\Documents\UCT\Maths and Applied Maths\MAM3040W\Project\mrs-processing\data\extracted\train"
hike_path = train_path + r"\hike.tags"


def create_mask(input, mask_value=0):
    mask = np.zeros(input.shape)
    mask = input != -1
    print(mask)
    return mask

def tag_word_separator(input):
    tokens = input.replace("[", " [ ").replace("]", " ] ").split()

    out_tokens = []
    out_tags = []

    index = 0

    while index < len(tokens):
        # get the tag
        if tokens[index] == "[":
            index += 1
            out_tags.append(tokens[index])
        index += 1
        # get the token
        if tokens[index] == "]":
            index += 1
            phrase = tokens[index]
            index += 1
            # make sure that we capture any phrases as well
            while index < len(tokens) and tokens[index] != "[":
                phrase += " " + tokens[index]
                index += 1
            out_tokens.append(phrase)

    return [out_tokens, out_tags]


def get_sentence_vector(sentence, word_to_id_dict):
    output = [word_to_id_dict[word] for word in sentence]
    return np.array(output)


def tag_distribution_vector(tag, tag_to_id_dict):
    output = np.zeros(len(tag_to_id_dict))
    output[tag_to_id_dict[tag]] = 1
    return output


def simple():
    # Read data
    train = []
    test = []
    dev = []

    with open(hike_path, "r+") as file:
        for line in file:
            train.append(tag_word_separator(line.strip()))


    # Set Up Indices
    # this allows us to move from a word to its index in the word_list and from an index to its word

    # this will correspond to the words from the sentences
    id_to_token = [] # put in the appropriate tokens here
    token_to_id = {} # set up the associated token to id index (must correspond with the id_to_token)

    # this will correspond to the span labels
    id_to_tag = []
    tag_to_id = {}

    for tokens, labels in train + test + dev:
        for token in tokens:
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
                id_to_token.append(token)

        for label in labels:
            if label not in tag_to_id:
                tag_to_id[label] = len(tag_to_id)
                id_to_tag.append(label)

    NWORDS = len(id_to_token)
    NTAGS = len(id_to_tag)

    # get GloVe embeddings
    pretrained = {}
    for line in open(GLOVE, "r", encoding="utf8"):
        parts = line.strip().split()
        word = parts[0]
        pretrained[word] = parts[1:]

    pretrained_list = []
    scale = np.sqrt(3.0 / DIM_EMBEDDING)
    for word in id_to_token:
        if word.lower() in pretrained:
            vector = [float(v) for v in pretrained[word.lower()]]
            pretrained_list.append(np.array(vector))
        else:
            random_vector = np.random.uniform(-scale, scale, [DIM_EMBEDDING])
            pretrained_list.append(random_vector)

    assert len(pretrained_list) == NWORDS

    # pretrained.__del__()

    # data input
    # sentence = "input from the data"

    # create the data in x and y format
    # x_train = []
    # y_train = []

    # for x, y in train:
    #     x_train.append(get_sentence_vector(x, token_to_id))
    #     y_train.append(get_gold_tag_vector(y, tag_to_id))

    for i in range(len(train)):
        tokens = train[i][0]
        tags = train[i][1]

        sequence_array = np.array(get_sentence_vector(tokens, token_to_id))

        tag_dist = np.array(
            [tag_distribution_vector(tag, tag_to_id) for tag in tags]
        )
        # convert the tags to the probability distribution form
        train[i] = [sequence_array, tag_dist]

    # x_test = []
    # y_test = []
    #
    # for x, y in test:
    #     x_test.append(get_sentence_vector(x, token_to_id))
    #     y_test.append(get_gold_tag_vector(y, tag_to_id))

    x_train = np.array([inp[0] for inp in train])
    # x_test = np.asarray(x_test)
    y_train = np.array([inp[1] for inp in train])
    # y_test = np.asarray(y_test)

    print(x_train.shape)
    print(y_train.shape)
    # print(x_test.shape)
    # print(y_test.shape)


    # # Padding and Masking batches
    x_train = keras.preprocessing.sequence.pad_sequences(
        x_train,
        value=-1
    )
    y_train = keras.preprocessing.sequence.pad_sequences(
        y_train,
        value=-1
    )

    # Mask the data
    # masker = layers.Masking(mask_value=-1)
    #
    # x_train = masker(x_train)
    # y_train = masker(y_train)
    #
    # print(x_train._keras_mask)
    # print(x_train.shape)


    # MODEL
    model = keras.Sequential()
    lstm_hidden = 64

    # Layers:

    #   Input — sequential
    # Define the inputs
    #
    #   Embedding — What dimension input vectors
    #             — embeddings will consist of a word vector and a PoS embedding concatenated (add this in later)
    # Layers to parse the sentence
    glove_init = keras.initializers.Constant(np.array(pretrained_list))
    model.add(layers.Embedding(
        input_dim=NWORDS,
        output_dim=DIM_EMBEDDING,
        trainable=False,
        embeddings_initializer=glove_init,
        # mask_zero=True
    ))

    # include the character level embeddings
    # probably use a cnn for this

    #   BLSTM — bidirectional LSTM
    #         — How many hidden layers and units
    #         — Only output after the whole span has been parsed, not at every input
    model.add(
        layers.Bidirectional(layers.LSTM(lstm_hidden, return_sequences=True), input_dim=DIM_EMBEDDING)
    )
    # Now we just need to get access to the lstm_output values to create the span encodings


    #   Dense — What does this do?
    #         — Will this actually just be the output layer?
    #         — Number of units
    model.add(layers.Dense(NTAGS))

    #   Output — A unit for every type of output label
    # model.add(layers.Dense(1, activation="softmax"))

    model.summary()

    # Define/choose loss function
    # Define/choose gradient method
    model.compile(
        loss=keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        optimizer="adam",
        metrics=["accuracy"]
    )

    model.fit(
        # x_train, y_train, validation_data=(x_test, y_test), epochs=1
        x_train, y_train, epochs=1
    )
    #
    #
    # TRAINING FUNCTION #


class TagParser(Model):
    def __init__(self, vocab_size, num_tags, pretrained_list, hidden_dim=32, embedding_dim=100):
        super(TagParser, self).__init__()

        # Layers to parse the sentence
        glove_init = keras.initializers.Constant(np.array(pretrained_list))

        self.Embed = layers.Embedding(vocab_size, embedding_dim, embeddings_initializer=glove_init)

        self.BSLTM = layers.Bidirectional(layers.LSTM(hidden_dim, return_sequences=True))

        self.Dense = layers.Dense(num_tags)

    def call(self, x, mask):
        pass


@tf.function
def train_step(sentences, tags):

    pass


@tf.function
def test_step(sentences, tags):
    pass

