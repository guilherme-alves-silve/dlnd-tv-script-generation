
# TV Script Generation
In this project, you'll generate your own [Simpsons](https://en.wikipedia.org/wiki/The_Simpsons) TV scripts using RNNs.  You'll be using part of the [Simpsons dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data) of scripts from 27 seasons.  The Neural Network you'll build will generate a new TV script for a scene at [Moe's Tavern](https://simpsonswiki.com/wiki/Moe's_Tavern).
## Get the Data
The data is already provided for you.  You'll be using a subset of the original dataset.  It consists of only the scenes in Moe's Tavern.  This doesn't include other versions of the tavern, like "Moe's Cavern", "Flaming Moe's", "Uncle Moe's Family Feed-Bag", etc..


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper

data_dir = './data/simpsons/moes_tavern_lines.txt'
text = helper.load_data(data_dir)
# Ignore notice, since we don't use it for analysing the data
text = text[81:]
```

## Explore the Data
Play around with `view_sentence_range` to view different parts of the data.


```python
view_sentence_range = (0, 10)

"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import numpy as np

print('Dataset Stats')
print('Roughly the number of unique words: {}'.format(len({word: None for word in text.split()})))
scenes = text.split('\n\n')
print('Number of scenes: {}'.format(len(scenes)))
sentence_count_scene = [scene.count('\n') for scene in scenes]
print('Average number of sentences in each scene: {}'.format(np.average(sentence_count_scene)))

sentences = [sentence for scene in scenes for sentence in scene.split('\n')]
print('Number of lines: {}'.format(len(sentences)))
word_count_sentence = [len(sentence.split()) for sentence in sentences]
print('Average number of words in each line: {}'.format(np.average(word_count_sentence)))

print()
print('The sentences {} to {}:'.format(*view_sentence_range))
print('\n'.join(text.split('\n')[view_sentence_range[0]:view_sentence_range[1]]))
```

    Dataset Stats
    Roughly the number of unique words: 11492
    Number of scenes: 262
    Average number of sentences in each scene: 15.251908396946565
    Number of lines: 4258
    Average number of words in each line: 11.50164396430249
    
    The sentences 0 to 10:
    
    Moe_Szyslak: (INTO PHONE) Moe's Tavern. Where the elite meet to drink.
    Bart_Simpson: Eh, yeah, hello, is Mike there? Last name, Rotch.
    Moe_Szyslak: (INTO PHONE) Hold on, I'll check. (TO BARFLIES) Mike Rotch. Mike Rotch. Hey, has anybody seen Mike Rotch, lately?
    Moe_Szyslak: (INTO PHONE) Listen you little puke. One of these days I'm gonna catch you, and I'm gonna carve my name on your back with an ice pick.
    Moe_Szyslak: What's the matter Homer? You're not your normal effervescent self.
    Homer_Simpson: I got my problems, Moe. Give me another one.
    Moe_Szyslak: Homer, hey, you should not drink to forget your problems.
    Barney_Gumble: Yeah, you should only drink to enhance your social skills.
    
    

## Implement Preprocessing Functions
The first thing to do to any dataset is preprocessing.  Implement the following preprocessing functions below:
- Lookup Table
- Tokenize Punctuation

### Lookup Table
To create a word embedding, you first need to transform the words to ids.  In this function, create two dictionaries:
- Dictionary to go from the words to an id, we'll call `vocab_to_int`
- Dictionary to go from the id to word, we'll call `int_to_vocab`

Return these dictionaries in the following tuple `(vocab_to_int, int_to_vocab)`


```python
import numpy as np
import problem_unittests as tests

def create_lookup_tables(text):
    """
    Create lookup tables for vocabulary
    :param text: The text of tv scripts split into words
    :return: A tuple of dicts (vocab_to_int, int_to_vocab)
    """
    text = set(text)
    vocab_to_int = {word: i for i, word in enumerate(text)}
    int_to_vocab = {i: word for word, i in vocab_to_int.items()}
    return vocab_to_int, int_to_vocab


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_create_lookup_tables(create_lookup_tables)
```

    Tests Passed
    

### Tokenize Punctuation
We'll be splitting the script into a word array using spaces as delimiters.  However, punctuations like periods and exclamation marks make it hard for the neural network to distinguish between the word "bye" and "bye!".

Implement the function `token_lookup` to return a dict that will be used to tokenize symbols like "!" into "||Exclamation_Mark||".  Create a dictionary for the following symbols where the symbol is the key and value is the token:
- Period ( . )
- Comma ( , )
- Quotation Mark ( " )
- Semicolon ( ; )
- Exclamation mark ( ! )
- Question mark ( ? )
- Left Parentheses ( ( )
- Right Parentheses ( ) )
- Dash ( -- )
- Return ( \n )

This dictionary will be used to token the symbols and add the delimiter (space) around it.  This separates the symbols as it's own word, making it easier for the neural network to predict on the next word. Make sure you don't use a token that could be confused as a word. Instead of using the token "dash", try using something like "||dash||".


```python
def token_lookup():
    """
    Generate a dict to turn punctuation into a token.
    :return: Tokenize dictionary where the key is the punctuation and the value is the token
    """
    tokens = {
        '.'  : '||period||' ,
        ','  : '||comma||' ,
        '"'  : '||quotation_mark||' ,
        ';'  : '||semicolon||' ,
        '!'  : '||exclamation_mark||' ,
        '?'  : '||question_mark||' ,
        '('  : '||left_parentheses||' ,
        ')'  : '||right_parentheses||' ,
        '--' : '||dash||' ,
        '\n' : '||return||'
    }
    return tokens

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_tokenize(token_lookup)
```

    Tests Passed
    

## Preprocess all the data and save it
Running the code cell below will preprocess all the data and save it to file.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Preprocess Training, Validation, and Testing Data
helper.preprocess_and_save_data(data_dir, token_lookup, create_lookup_tables)
```

# Check Point
This is your first checkpoint. If you ever decide to come back to this notebook or have to restart the notebook, you can start from here. The preprocessed data has been saved to disk.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import helper
import numpy as np
import problem_unittests as tests

int_text, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
```

## Build the Neural Network
You'll build the components necessary to build a RNN by implementing the following functions below:
- get_inputs
- get_init_cell
- get_embed
- build_rnn
- build_nn
- get_batches

### Check the Version of TensorFlow and Access to GPU


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from distutils.version import LooseVersion
import warnings
import tensorflow as tf

# Check TensorFlow Version
assert LooseVersion(tf.__version__) >= LooseVersion('1.0'), 'Please use TensorFlow version 1.0 or newer'
print('TensorFlow Version: {}'.format(tf.__version__))

# Check for a GPU
if not tf.test.gpu_device_name():
    warnings.warn('No GPU found. Please use a GPU to train your neural network.')
else:
    print('Default GPU Device: {}'.format(tf.test.gpu_device_name()))
```

    TensorFlow Version: 1.0.0
    

    C:\Users\Kurosaki-X\Anaconda3\envs\dlnd-tf-lab\lib\site-packages\ipykernel\__main__.py:14: UserWarning: No GPU found. Please use a GPU to train your neural network.
    

### Input
Implement the `get_inputs()` function to create TF Placeholders for the Neural Network.  It should create the following placeholders:
- Input text placeholder named "input" using the [TF Placeholder](https://www.tensorflow.org/api_docs/python/tf/placeholder) `name` parameter.
- Targets placeholder
- Learning Rate placeholder

Return the placeholders in the following tuple `(Input, Targets, LearningRate)`


```python
def get_inputs():
    """
    Create TF Placeholders for input, targets, and learning rate.
    :return: Tuple (input, targets, learning rate)
    """
    inputs = tf.placeholder(tf.int32, [None, None], name='input')
    targets = tf.placeholder(tf.int32, [None, None], name='targets')
    learning_rate = tf.placeholder(tf.float32, name='learning_rate')
    return inputs, targets, learning_rate


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_inputs(get_inputs)
```

    Tests Passed
    

### Build RNN Cell and Initialize
Stack one or more [`BasicLSTMCells`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/BasicLSTMCell) in a [`MultiRNNCell`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell).
- The Rnn size should be set using `rnn_size`
- Initalize Cell State using the MultiRNNCell's [`zero_state()`](https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/MultiRNNCell#zero_state) function
    - Apply the name "initial_state" to the initial state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the cell and initial state in the following tuple `(Cell, InitialState)`


```python
def get_init_cell(batch_size, rnn_size, keep_prob=None, num_cells=1):
    """
    Create an RNN Cell and initialize it.
    :param batch_size: Size of batches
    :param rnn_size: Size of RNNs
    :return: Tuple (cell, initialize state)
    """

    def build_cell(rnn_size, keep_prob):
        lstm = tf.contrib.rnn.BasicLSTMCell(rnn_size)
        if keep_prob is not None:
            return tf.contrib.rnn.DropoutWrapper(lstm, output_keep_prob=keep_prob)
        else:
            return lstm
    
    cell = tf.contrib.rnn.MultiRNNCell([build_cell(rnn_size, keep_prob) for _ in range(num_cells)])
    initial_state = tf.identity(cell.zero_state(batch_size, tf.float32), name='initial_state')
    
    return cell, initial_state


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_init_cell(get_init_cell)
```

    Tests Passed
    

### Word Embedding
Apply embedding to `input_data` using TensorFlow.  Return the embedded sequence.


```python
def get_embed(input_data, vocab_size, embed_dim):
    """
    Create embedding for <input_data>.
    :param input_data: TF placeholder for text input.
    :param vocab_size: Number of words in vocabulary.
    :param embed_dim: Number of embedding dimensions
    :return: Embedded input.
    """
    #random_uniform even if it's closer to the mean, doesn't work well, it become worse
    #embedding = tf.Variable(tf.random_uniform([vocab_size, embed_dim], -0.2, 0.2), name='embedding')
    #Suggested in the review to use a distribution between -0.2 and 0.2 because this data is closer to the mean, but
    #0.1 works better in my configuration
    embedding = tf.Variable(tf.truncated_normal([vocab_size, embed_dim], stddev=0.1), dtype=tf.float32, name='embedding')
    embed = tf.nn.embedding_lookup(embedding, input_data, name='embed')
    
    return embed


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_embed(get_embed)
```

    Tests Passed
    

### Build RNN
You created a RNN Cell in the `get_init_cell()` function.  Time to use the cell to create a RNN.
- Build the RNN using the [`tf.nn.dynamic_rnn()`](https://www.tensorflow.org/api_docs/python/tf/nn/dynamic_rnn)
 - Apply the name "final_state" to the final state using [`tf.identity()`](https://www.tensorflow.org/api_docs/python/tf/identity)

Return the outputs and final_state state in the following tuple `(Outputs, FinalState)` 


```python
def build_rnn(cell, inputs):
    """
    Create a RNN using a RNN Cell
    :param cell: RNN Cell
    :param inputs: Input text data
    :return: Tuple (Outputs, Final State)
    """
    output, final_state = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32)
    final_state = tf.identity(final_state, name='final_state')
    return output, final_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_rnn(build_rnn)
```

    Tests Passed
    

### Build the Neural Network
Apply the functions you implemented above to:
- Apply embedding to `input_data` using your `get_embed(input_data, vocab_size, embed_dim)` function.
- Build RNN using `cell` and your `build_rnn(cell, inputs)` function.
- Apply a fully connected layer with a linear activation and `vocab_size` as the number of outputs.

Return the logits and final state in the following tuple (Logits, FinalState) 


```python
def build_nn(cell, rnn_size, input_data, vocab_size, embed_dim):
    """
    Build part of the neural network
    :param cell: RNN cell
    :param rnn_size: Size of rnns
    :param input_data: Input data
    :param vocab_size: Vocabulary size
    :param embed_dim: Number of embedding dimensions
    :return: Tuple (Logits, FinalState)
    """
    embed = get_embed(input_data, vocab_size, embed_dim)
    output, final_state = build_rnn(cell, embed)
    #Activation None is the same as linear activation
    output = tf.contrib.layers.fully_connected(output, vocab_size, activation_fn=None)
    return output, final_state

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_build_nn(build_nn)
```

    Tests Passed
    

### Batches
Implement `get_batches` to create batches of input and targets using `int_text`.  The batches should be a Numpy array with the shape `(number of batches, 2, batch size, sequence length)`. Each batch contains two elements:
- The first element is a single batch of **input** with the shape `[batch size, sequence length]`
- The second element is a single batch of **targets** with the shape `[batch size, sequence length]`

If you can't fill the last batch with enough data, drop the last batch.

For example, `get_batches([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20], 3, 2)` would return a Numpy array of the following:
```
[
  # First Batch
  [
    # Batch of Input
    [[ 1  2], [ 7  8], [13 14]]
    # Batch of targets
    [[ 2  3], [ 8  9], [14 15]]
  ]

  # Second Batch
  [
    # Batch of Input
    [[ 3  4], [ 9 10], [15 16]]
    # Batch of targets
    [[ 4  5], [10 11], [16 17]]
  ]

  # Third Batch
  [
    # Batch of Input
    [[ 5  6], [11 12], [17 18]]
    # Batch of targets
    [[ 6  7], [12 13], [18  1]]
  ]
]
```

Notice that the last target value in the last batch is the first input value of the first batch. In this case, `1`. This is a common technique used when creating sequence batches, although it is rather unintuitive.


```python
def get_batches(int_text, batch_size, seq_length):
    """
    Return batches of input and target
    :param int_text: Text with the words replaced by their ids
    :param batch_size: The size of batch
    :param seq_length: The length of sequence
    :return: Batches as a Numpy array
    """
    n_batches = len(int_text) // batch_size // seq_length
    n_chars = n_batches * seq_length
    full_x = int_text[0: n_chars * batch_size]
    full_y = int_text[1: n_chars * batch_size + 1]
    full_y[-1] = full_x[0]
        
    part_x = []
    part_y = []
    for batch in range(batch_size):
        start = batch * n_chars
        end = start + (n_chars)
        part_x.append(full_x[start:end])
        part_y.append(full_y[start:end])
    
    split_x = np.split(np.array(part_x), n_batches,1)
    split_y = np.split(np.array(part_y), n_batches,1)
    
    batches = []
    for batch_x, batch_y in zip(split_x, split_y):
        batches.append([batch_x, batch_y])
    
    return np.array(batches)

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_batches(get_batches)
```

    Tests Passed
    

## Neural Network Training
### Hyperparameters
Tune the following parameters:

- Set `num_epochs` to the number of epochs.
- Set `batch_size` to the batch size.
- Set `rnn_size` to the size of the RNNs.
- Set `embed_dim` to the size of the embedding.
- Set `seq_length` to the length of sequence.
- Set `learning_rate` to the learning rate.
- Set `show_every_n_batches` to the number of batches the neural network should print progress.


```python
# Number of Epochs
num_epochs = 150
# Batch Size
batch_size = 512
# RNN Size
rnn_size = 260
# Embedding Dimension Size
embed_dim = 250
# Sequence Length
seq_length = 35
# Learning Rate
learning_rate = 0.01
# Show stats for every n number of batches
show_every_n_batches = 3
#Number of neurons to keep in the layers
keep_probability = 0.9

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
save_dir = './save'
```

### Build the Graph
Build the graph using the neural network you implemented.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
from tensorflow.contrib import seq2seq

train_graph = tf.Graph()
with train_graph.as_default():
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    vocab_size = len(int_to_vocab)
    input_text, targets, lr = get_inputs()
    input_data_shape = tf.shape(input_text)
    cell, initial_state = get_init_cell(input_data_shape[0], rnn_size, keep_prob)
    logits, final_state = build_nn(cell, rnn_size, input_text, vocab_size, embed_dim)

    # Probabilities for generating words
    probs = tf.nn.softmax(logits, name='probs')

    # Loss function
    cost = seq2seq.sequence_loss(
        logits,
        targets,
        tf.ones([input_data_shape[0], input_data_shape[1]]))

    # Optimizer
    optimizer = tf.train.AdamOptimizer(lr)

    # Gradient Clipping
    gradients = optimizer.compute_gradients(cost)
    capped_gradients = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gradients if grad is not None]
    train_op = optimizer.apply_gradients(capped_gradients)
```

## Train
Train the neural network on the preprocessed data.  If you have a hard time getting a good loss, check the [forums](https://discussions.udacity.com/) to see if anyone is having the same problem.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
batches = get_batches(int_text, batch_size, seq_length)

with tf.Session(graph=train_graph) as sess:
    sess.run(tf.global_variables_initializer())

    for epoch_i in range(num_epochs):
        state = sess.run(initial_state, {input_text: batches[0][0]})

        for batch_i, (x, y) in enumerate(batches):
            feed = {
                input_text: x,
                targets: y,
                initial_state: state,
                lr: learning_rate,
                keep_prob: keep_probability}
            train_loss, state, _ = sess.run([cost, final_state, train_op], feed)

            # Show every <show_every_n_batches> batches
            if (epoch_i * len(batches) + batch_i) % show_every_n_batches == 0:
                print('Epoch {:>3} Batch {:>4}/{}   train_loss = {:.3f}'.format(
                    epoch_i,
                    batch_i,
                    len(batches),
                    train_loss))

    # Save Model
    saver = tf.train.Saver()
    saver.save(sess, save_dir)
    print('Model Trained and Saved')
```

    Epoch   0 Batch    0/3   train_loss = 8.822
    Epoch   1 Batch    0/3   train_loss = 6.505
    Epoch   2 Batch    0/3   train_loss = 6.237
    Epoch   3 Batch    0/3   train_loss = 6.044
    Epoch   4 Batch    0/3   train_loss = 5.967
    Epoch   5 Batch    0/3   train_loss = 5.934
    Epoch   6 Batch    0/3   train_loss = 5.803
    Epoch   7 Batch    0/3   train_loss = 5.696
    Epoch   8 Batch    0/3   train_loss = 5.576
    Epoch   9 Batch    0/3   train_loss = 5.450
    Epoch  10 Batch    0/3   train_loss = 5.325
    Epoch  11 Batch    0/3   train_loss = 5.219
    Epoch  12 Batch    0/3   train_loss = 5.120
    Epoch  13 Batch    0/3   train_loss = 5.014
    Epoch  14 Batch    0/3   train_loss = 4.908
    Epoch  15 Batch    0/3   train_loss = 4.800
    Epoch  16 Batch    0/3   train_loss = 4.690
    Epoch  17 Batch    0/3   train_loss = 4.585
    Epoch  18 Batch    0/3   train_loss = 4.493
    Epoch  19 Batch    0/3   train_loss = 4.403
    Epoch  20 Batch    0/3   train_loss = 4.316
    Epoch  21 Batch    0/3   train_loss = 4.232
    Epoch  22 Batch    0/3   train_loss = 4.152
    Epoch  23 Batch    0/3   train_loss = 4.072
    Epoch  24 Batch    0/3   train_loss = 3.995
    Epoch  25 Batch    0/3   train_loss = 3.919
    Epoch  26 Batch    0/3   train_loss = 3.850
    Epoch  27 Batch    0/3   train_loss = 3.778
    Epoch  28 Batch    0/3   train_loss = 3.711
    Epoch  29 Batch    0/3   train_loss = 3.652
    Epoch  30 Batch    0/3   train_loss = 3.584
    Epoch  31 Batch    0/3   train_loss = 3.533
    Epoch  32 Batch    0/3   train_loss = 3.462
    Epoch  33 Batch    0/3   train_loss = 3.405
    Epoch  34 Batch    0/3   train_loss = 3.339
    Epoch  35 Batch    0/3   train_loss = 3.282
    Epoch  36 Batch    0/3   train_loss = 3.223
    Epoch  37 Batch    0/3   train_loss = 3.166
    Epoch  38 Batch    0/3   train_loss = 3.114
    Epoch  39 Batch    0/3   train_loss = 3.059
    Epoch  40 Batch    0/3   train_loss = 3.017
    Epoch  41 Batch    0/3   train_loss = 2.966
    Epoch  42 Batch    0/3   train_loss = 2.929
    Epoch  43 Batch    0/3   train_loss = 2.866
    Epoch  44 Batch    0/3   train_loss = 2.833
    Epoch  45 Batch    0/3   train_loss = 2.782
    Epoch  46 Batch    0/3   train_loss = 2.730
    Epoch  47 Batch    0/3   train_loss = 2.698
    Epoch  48 Batch    0/3   train_loss = 2.644
    Epoch  49 Batch    0/3   train_loss = 2.616
    Epoch  50 Batch    0/3   train_loss = 2.574
    Epoch  51 Batch    0/3   train_loss = 2.528
    Epoch  52 Batch    0/3   train_loss = 2.490
    Epoch  53 Batch    0/3   train_loss = 2.457
    Epoch  54 Batch    0/3   train_loss = 2.415
    Epoch  55 Batch    0/3   train_loss = 2.389
    Epoch  56 Batch    0/3   train_loss = 2.351
    Epoch  57 Batch    0/3   train_loss = 2.321
    Epoch  58 Batch    0/3   train_loss = 2.278
    Epoch  59 Batch    0/3   train_loss = 2.257
    Epoch  60 Batch    0/3   train_loss = 2.227
    Epoch  61 Batch    0/3   train_loss = 2.191
    Epoch  62 Batch    0/3   train_loss = 2.153
    Epoch  63 Batch    0/3   train_loss = 2.127
    Epoch  64 Batch    0/3   train_loss = 2.106
    Epoch  65 Batch    0/3   train_loss = 2.078
    Epoch  66 Batch    0/3   train_loss = 2.056
    Epoch  67 Batch    0/3   train_loss = 2.033
    Epoch  68 Batch    0/3   train_loss = 2.002
    Epoch  69 Batch    0/3   train_loss = 1.979
    Epoch  70 Batch    0/3   train_loss = 1.939
    Epoch  71 Batch    0/3   train_loss = 1.919
    Epoch  72 Batch    0/3   train_loss = 1.889
    Epoch  73 Batch    0/3   train_loss = 1.869
    Epoch  74 Batch    0/3   train_loss = 1.833
    Epoch  75 Batch    0/3   train_loss = 1.816
    Epoch  76 Batch    0/3   train_loss = 1.798
    Epoch  77 Batch    0/3   train_loss = 1.779
    Epoch  78 Batch    0/3   train_loss = 1.745
    Epoch  79 Batch    0/3   train_loss = 1.757
    Epoch  80 Batch    0/3   train_loss = 1.716
    Epoch  81 Batch    0/3   train_loss = 1.715
    Epoch  82 Batch    0/3   train_loss = 1.672
    Epoch  83 Batch    0/3   train_loss = 1.676
    Epoch  84 Batch    0/3   train_loss = 1.644
    Epoch  85 Batch    0/3   train_loss = 1.612
    Epoch  86 Batch    0/3   train_loss = 1.615
    Epoch  87 Batch    0/3   train_loss = 1.574
    Epoch  88 Batch    0/3   train_loss = 1.568
    Epoch  89 Batch    0/3   train_loss = 1.552
    Epoch  90 Batch    0/3   train_loss = 1.527
    Epoch  91 Batch    0/3   train_loss = 1.533
    Epoch  92 Batch    0/3   train_loss = 1.497
    Epoch  93 Batch    0/3   train_loss = 1.505
    Epoch  94 Batch    0/3   train_loss = 1.463
    Epoch  95 Batch    0/3   train_loss = 1.461
    Epoch  96 Batch    0/3   train_loss = 1.429
    Epoch  97 Batch    0/3   train_loss = 1.416
    Epoch  98 Batch    0/3   train_loss = 1.392
    Epoch  99 Batch    0/3   train_loss = 1.372
    Epoch 100 Batch    0/3   train_loss = 1.358
    Epoch 101 Batch    0/3   train_loss = 1.348
    Epoch 102 Batch    0/3   train_loss = 1.333
    Epoch 103 Batch    0/3   train_loss = 1.322
    Epoch 104 Batch    0/3   train_loss = 1.296
    Epoch 105 Batch    0/3   train_loss = 1.297
    Epoch 106 Batch    0/3   train_loss = 1.278
    Epoch 107 Batch    0/3   train_loss = 1.277
    Epoch 108 Batch    0/3   train_loss = 1.260
    Epoch 109 Batch    0/3   train_loss = 1.272
    Epoch 110 Batch    0/3   train_loss = 1.239
    Epoch 111 Batch    0/3   train_loss = 1.250
    Epoch 112 Batch    0/3   train_loss = 1.236
    Epoch 113 Batch    0/3   train_loss = 1.211
    Epoch 114 Batch    0/3   train_loss = 1.189
    Epoch 115 Batch    0/3   train_loss = 1.173
    Epoch 116 Batch    0/3   train_loss = 1.158
    Epoch 117 Batch    0/3   train_loss = 1.146
    Epoch 118 Batch    0/3   train_loss = 1.132
    Epoch 119 Batch    0/3   train_loss = 1.122
    Epoch 120 Batch    0/3   train_loss = 1.103
    Epoch 121 Batch    0/3   train_loss = 1.111
    Epoch 122 Batch    0/3   train_loss = 1.091
    Epoch 123 Batch    0/3   train_loss = 1.074
    Epoch 124 Batch    0/3   train_loss = 1.086
    Epoch 125 Batch    0/3   train_loss = 1.058
    Epoch 126 Batch    0/3   train_loss = 1.078
    Epoch 127 Batch    0/3   train_loss = 1.037
    Epoch 128 Batch    0/3   train_loss = 1.044
    Epoch 129 Batch    0/3   train_loss = 1.017
    Epoch 130 Batch    0/3   train_loss = 1.012
    Epoch 131 Batch    0/3   train_loss = 1.002
    Epoch 132 Batch    0/3   train_loss = 0.994
    Epoch 133 Batch    0/3   train_loss = 0.980
    Epoch 134 Batch    0/3   train_loss = 0.986
    Epoch 135 Batch    0/3   train_loss = 0.962
    Epoch 136 Batch    0/3   train_loss = 0.965
    Epoch 137 Batch    0/3   train_loss = 0.973
    Epoch 138 Batch    0/3   train_loss = 0.957
    Epoch 139 Batch    0/3   train_loss = 0.950
    Epoch 140 Batch    0/3   train_loss = 0.943
    Epoch 141 Batch    0/3   train_loss = 0.932
    Epoch 142 Batch    0/3   train_loss = 0.923
    Epoch 143 Batch    0/3   train_loss = 0.908
    Epoch 144 Batch    0/3   train_loss = 0.910
    Epoch 145 Batch    0/3   train_loss = 0.893
    Epoch 146 Batch    0/3   train_loss = 0.893
    Epoch 147 Batch    0/3   train_loss = 0.878
    Epoch 148 Batch    0/3   train_loss = 0.869
    Epoch 149 Batch    0/3   train_loss = 0.854
    Model Trained and Saved
    

## Save Parameters
Save `seq_length` and `save_dir` for generating a new TV script.


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
# Save parameters for checkpoint
helper.save_params((seq_length, save_dir))
```

# Checkpoint


```python
"""
DON'T MODIFY ANYTHING IN THIS CELL
"""
import tensorflow as tf
import numpy as np
import helper
import problem_unittests as tests

_, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
seq_length, load_dir = helper.load_params()
```

## Implement Generate Functions
### Get Tensors
Get tensors from `loaded_graph` using the function [`get_tensor_by_name()`](https://www.tensorflow.org/api_docs/python/tf/Graph#get_tensor_by_name).  Get the tensors using the following names:
- "input:0"
- "initial_state:0"
- "final_state:0"
- "probs:0"

Return the tensors in the following tuple `(InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)` 


```python
def get_tensors(loaded_graph):
    """
    Get input, initial state, final state, and probabilities tensor from <loaded_graph>
    :param loaded_graph: TensorFlow graph loaded from file
    :return: Tuple (InputTensor, InitialStateTensor, FinalStateTensor, ProbsTensor)
    """
    # TODO: Implement Function
    return loaded_graph.get_tensor_by_name("input:0"), loaded_graph.get_tensor_by_name("initial_state:0"), loaded_graph.get_tensor_by_name("final_state:0"), loaded_graph.get_tensor_by_name("probs:0")


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_get_tensors(get_tensors)
```

    Tests Passed
    

### Choose Word
Implement the `pick_word()` function to select the next word using `probabilities`.


```python
def pick_word(probabilities, int_to_vocab):
    """
    Pick the next word in the generated text
    :param probabilities: Probabilites of the next word
    :param int_to_vocab: Dictionary of word ids as the keys and words as the values
    :return: String of the predicted word
    """
    #https://stackoverflow.com/questions/3679694/a-weighted-version-of-random-choice
    idx = np.random.choice(len(int_to_vocab), p=probabilities)
    return int_to_vocab[idx]


"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
tests.test_pick_word(pick_word)
```

    Tests Passed
    

## Generate TV Script
This will generate the TV script for you.  Set `gen_length` to the length of TV script you want to generate.


```python
gen_length = 200
# homer_simpson, moe_szyslak, or Barney_Gumble
prime_word = 'homer_simpson'

"""
DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
"""
loaded_graph = tf.Graph()
with tf.Session(graph=loaded_graph) as sess:
    # Load saved model
    loader = tf.train.import_meta_graph(load_dir + '.meta')
    loader.restore(sess, load_dir)

    # Get Tensors from loaded model
    input_text, initial_state, final_state, probs = get_tensors(loaded_graph)
    keep_prob = loaded_graph.get_tensor_by_name('keep_prob:0')

    # Sentences generation setup
    gen_sentences = [prime_word + ':']
    prev_state = sess.run(initial_state, {input_text: np.array([[1]]), keep_prob: 1.0})

    # Generate sentences
    for n in range(gen_length):
        # Dynamic Input
        dyn_input = [[vocab_to_int[word] for word in gen_sentences[-seq_length:]]]
        dyn_seq_length = len(dyn_input[0])

        # Get Prediction
        probabilities, prev_state = sess.run(
            [probs, final_state],
            {input_text: dyn_input, initial_state: prev_state, keep_prob: 1.0})
        
        pred_word = pick_word(probabilities[dyn_seq_length-1], int_to_vocab)

        gen_sentences.append(pred_word)
    
    # Remove tokens
    tv_script = ' '.join(gen_sentences)
    for key, token in token_dict.items():
        ending = ' ' if key in ['\n', '(', '"'] else ''
        tv_script = tv_script.replace(' ' + token.lower(), key)
    tv_script = tv_script.replace('\n ', '\n')
    tv_script = tv_script.replace('( ', '(')
        
    print(tv_script)
```

    homer_simpson: wooooo! 'topes ruuuule!
    ned_flanders: homer, moe, hurry!
    homer_simpson: moe, no problem.
    
    
    moe_szyslak: shove on...
    moe_szyslak: geez, homer...
    jacques: such a guy.
    lenny_leonard:(nods) mm, i never trusted her.
    lenny_leonard: don't forget that fish snout.
    moe_szyslak:(nasty laugh) yeah.(spoken) so whaddaya think, i'll just using it backward.
    barney_gumble: but who'll run the bar while i'm just like a full-time stock market guy.
    moe_szyslak: yeah. but i'm so desperately lonely.
    chief_wiggum: well, i need help you...
    moe_szyslak: it's a snap when you use certified contractors.
    c.
    homer_simpson:(absentmindedly going to keep it through playoff season.
    moe_szyslak: whoa, tha.....
    moe_szyslak: geez, homer. you've got a time for me.
    
    
    homer_simpson: ahhh, this was about that renders old people tolerable to us normals?
    lisa_simpson: that one?
    lenny_leonard: wow, i look
    

# The TV Script is Nonsensical
It's ok if the TV script doesn't make any sense.  We trained on less than a megabyte of text.  In order to get good results, you'll have to use a smaller vocabulary or get more data.  Luckly there's more data!  As we mentioned in the begging of this project, this is a subset of [another dataset](https://www.kaggle.com/wcukierski/the-simpsons-by-the-data).  We didn't have you train on all the data, because that would take too long.  However, you are free to train your neural network on all the data.  After you complete the project, of course.
# Submitting This Project
When submitting this project, make sure to run all the cells before saving the notebook. Save the notebook file as "dlnd_tv_script_generation.ipynb" and save it as a HTML file under "File" -> "Download as". Include the "helper.py" and "problem_unittests.py" files in your submission.
