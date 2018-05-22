# **Deep_Learning-(RNN)-based-Language_Translation_Machine** 

---

**Deep Learning (RNN) based Language Translation Machine**

The goals / steps of this project are the following:
* Use the Sequence to Sequnce model to make language translation
* First preprocessing the text data
* Build the sequence-to-sequence model to train with preprocessed data
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/RNN_models.jpg
[image2]: ./examples/S2S.jpg
[image3]: ./examples/words.jpg
[image4]: ./examples/S2S_2.jpg
[image5]: ./examples/decoder.jpg

### Data Exploration and Preprocessing

#### 1. Explore the Data

In the cell wth title `Explore the Data`, I show the dataset status like below:

```python
Dataset Stats
Roughly the number of unique words: 227
Number of sentences: 137861
Average number of words in a sentence: 13.225277634719028
```

And some sentences from dataset are like:

```
English sentences 0 to 10:
new jersey is sometimes quiet during autumn , and it is snowy in april .
the united states is usually chilly during july , and it is usually freezing in november .
california is usually quiet during march , and it is usually hot in june .
the united states is sometimes mild during june , and it is cold in september .
your least liked fruit is the grape , but my least liked is the apple .
his favorite fruit is the orange , but my favorite is the grape .
paris is relaxing during december , but it is usually chilly in july .
new jersey is busy during spring , and it is never hot in march .
our least liked fruit is the lemon , but my least liked is the grape .
the united states is sometimes busy during january , and it is sometimes warm in november .

French sentences 0 to 10:
new jersey est parfois calme pendant l' automne , et il est neigeux en avril .
les états-unis est généralement froid en juillet , et il gèle habituellement en novembre .
california est généralement calme en mars , et il est généralement chaud en juin .
les états-unis est parfois légère en juin , et il fait froid en septembre .
votre moins aimé fruit est le raisin , mais mon moins aimé est la pomme .
son fruit préféré est l'orange , mais mon préféré est le raisin .
paris est relaxant en décembre , mais il est généralement froid en juillet .
new jersey est occupé au printemps , et il est jamais chaude en mars .
notre fruit est moins aimé le citron , mais mon moins aimé est le raisin .
les états-unis est parfois occupé en janvier , et il est parfois chaud en novembre .

```

#### 2. Data Preprocessing

After showing the basic information of dataset, I make the data preprocessing:

```python 
Text to Word Ids
```

As I did with other RNNs, I must turn the text into a number so the computer can understand it. In the function text_to_ids(), I'll turn source_text and target_text from words to ids. However, I need to add the <EOS> word id at the end of target_text. This will help the neural network predict when the sentence should end.
  
I can get the <EOS> word id by doing:
  
```python 
target_vocab_to_int['<EOS>']
```

I can get other word ids using source_vocab_to_int and target_vocab_to_int.

After that, preprocess all data and save them.

### Build RNN

### 1. Build RNN Cell and Initialize

The RNN models are normally looks like following:

![alt text][image1]

where many to one represents that the inputs of RNN are many, the ouput is only one element, this normally used in sentiment analysis tasks. And many to many is a common method to build the language translation machine. 

The Sequence to Sequnce model can be simplized as following:

![alt text][image2]

we got 2 RNNs, one received the input sequences, then hands over what it has learned to the second RNN, which start to produce the output sequence. 

Supposing we have such input sequence, we want to translate it to target.
![alt text][image3]

So our encode and decode process looks like:
![alt text][image4]
At final step of decode, we will add fully connected layer, to specify which words should be the most likely output.


#### 2. Word Embedding
When you're dealing with words in text, you end up with tens of thousands of classes to predict, one for each word. Trying to one-hot encode these words is massively inefficient, you'll have one element set to 1 and the other 50,000 set to 0. The matrix multiplication going into the first hidden layer will have almost all of the resulting values be zero. This a huge waste of computation.



To solve this problem and greatly increase the efficiency of our networks, we use what are called embeddings. Embeddings are just a fully connected layer like you've seen before. We call this layer the embedding layer and the weights are embedding weights. We skip the multiplication into the embedding layer by instead directly grabbing the hidden layer values from the weight matrix. We can do this because the multiplication of a one-hot encoded vector with a matrix returns the row of the matrix corresponding the index of the "on" input unit.


Instead of doing the matrix multiplication, we use the weight matrix as a lookup table. We encode the words as integers, for example "heart" is encoded as 958, "mind" as 18094. Then to get hidden layer values for "heart", you just take the 958th row of the embedding matrix. This process is called an embedding lookup and the number of hidden units is the embedding dimension.


Embeddings aren't only used for words of course. You can use them for any model where you have a massive number of classes. A particular type of model called Word2Vec uses the embedding layer to find vector representations of words that contain semantic meaning.

Function `get_embed`: Apply embedding to input_data using TensorFlow. Return the embedded sequence. 

#### 3. Build RNN


### Neural Network Training

#### 1. Hyperparameters

I used the following hyperparameters to train my network, these parameters are chosen empirically and experimentally:

```python
# Number of Epochs
epochs = 10
# Batch Size
batch_size = 128
# RNN Size
rnn_size = 256
# Number of Layers
num_layers = 2
# Embedding Size
encoding_embedding_size = 256
decoding_embedding_size = 256
# Learning Rate
learning_rate = 0.001
# Dropout Keep Probability
keep_probability = 0.5
display_step = 30
```

### Test the trained model

Sentence Translation:

```
Input
  Word Ids:      [222, 173, 219, 92, 228, 198, 142]
  English Words: ['he', 'saw', 'a', 'old', 'yellow', 'truck', '.']

Prediction
  Word Ids:      [279, 265, 267, 111, 318, 159, 30, 234, 1]
  French Words: il a vu un vieux camion janvier . <EOS>
```




### Summary

The Tanslation Machine is not used for any sentences until now, because the dataset I used is comparably small, but what import is that the algorithms are right, with better GPU and large amount dataset to train the model, the machine will do better.
You can train on the [WMT10 French-English corpus](http://www.statmt.org/wmt10/training-giga-fren.tar).  This dataset has more vocabulary and richer in topics discussed.
