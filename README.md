## Machine translation program using deep learning learn
I used Tensorflow with the Keras API to create this fascinating app.

### Quick Summary of Tensorflow & LSTM models:
-   seq2seq models are deep learning models that use recurrent neural networks (RNNs) like LSTMs to generate output
-   In machine translation, seq2seq networks encompass two main parts:
    -   The encoder accepts language as input - outputting state vectors
    -   The decoder accepts the encoder’s final state - outputting possible translations
-   'Teacher forcing' is a method we use to train seq2seq decoders
-   We mark the beginning and end of target sentences so that the decoder knows what to expect at the beginning and end of sentences
-   'One-hot' vectors are a way to represent a given word in a set of words wherein we use '1' to indicate the current word and '0' to indicate every other word
-   'Timesteps' help us keep track of where we are in a sentence
-   We can adjust batch size, which determines how many sentences we give a model at a time
-   We can also tweak dimensionality and number of epochs, which can improve results with careful tuning
-   The 'Softmax' function converts the output of the LSTMs into a probability distribution over words in our vocabulary.

### Sample output on first attempt:
<p align="center"><img src="https://i.ibb.co/8DvyNxw/Tensorflow-output.jpg" alt="Tensorflow-output" border="0"></p>
