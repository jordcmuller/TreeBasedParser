import tensorflow as tf
from tensorflow.keras.layers import Bidirectional, Embedding, LSTM, Dense
from tensorflow.keras import Model
from tensorflow.keras.initializers import Constant
import numpy as np

DIM_EMBEDDING = 100

# train_data =
# test_data =


# Data Formatting
#
# input:
#       sentence
# gold:
#       number of the token that is being split on
#       This will probably be represented as a one-hot array over the tokens#


# The class for the neural parser.
class TreeParser(Model):
    def __init__(self, vocab_size, num_labels, pretrained_list, hidden_dim=32, embedding_dim=100):
        super(TreeParser, self).__init__()

        # Layers to parse the sentence
        glove_init = Constant(np.array(pretrained_list))

        self.Embed = Embedding(vocab_size, embedding_dim, embeddings_initializer=glove_init)

        self.LeftLSTM = LSTM(hidden_dim, return_sequences=True)
        self.RightLSTM = LSTM(hidden_dim, return_sequences=True)
        self.BLSTM = Bidirectional(self.LeftLSTM, backward_layer=self.RightLSTM)

        # Layers to parse the tree structure
        # The idea is for the input to be binary (taking the BLSTM sentence encoding and the two token or edge tags that
        # the span is over. this will be the vector [word vector, label1, label2]#
        self.Dense = Dense(num_labels)

        # argmax at some point (this won't be a tensorflow layer - not part of training - but rather a simple deterministic function)


    def call(self, x):
        size = len(x)
        CKY_table = np.zeros((size, size))

        x = self.Embed(x)
        x = self.BLSTM(x)

        for i in range(size):
            for j in range(size):


        pass



vocab_size = 100
label_size = 100

parser = TreeParser(vocab_size, label_size)

def train_step(sentence, labels):
    with tf.GradientTape() as tape:
        predictions = parser(sentence, training=true)
        pass
