# This will hold the model(s) for the label and tree parsers

# An important feature of these models is that they compute the forward and backward passes for an input sentence
# and then distribute these values in order to create the span encoding that is required for the parser.

# maybe it would be a good idea to extend this class from another tensorflow class??
class Model:
    def __init__(self):
        self.forward = None
        self.backward = None

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

    # The forward and backward LSTM units allow access to the forward and backwards weights so that we
    # can get the span encodings
    forward = layers.LSTM(64)
    backward = layers.LSTM(64)

    model.add(
        layers.Bidirectional(forward, backward_layer=backward, input_dim=DIM_EMBEDDING)
    )

    #   Dense — What does this do?
    #         — Will this actually just be the output layer?
    #         — Number of units
    model.add(layers.Dense(NTAGS))
