from __future__ import unicode_literals, print_function, division
import re
import unicodedata
from unittest.util import _MAX_LENGTH
import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import random
import numpy as np
import string

### Helper class for word indexing
SOS_TOKEN = 0 # Start of sentence
EOS_TOKEN = 1 # End of sentence
MAX_LENGTH = 26

# Let's define a QA (Questions/Answers) class
# since each class has its own 'language'.

class QA_Lang:
    """ 
    # The constructor should be specified by its:
    # - word2index, a dictionary that maps each word to each index
    # - index2word, a dictionary that maps each index to each word
    # - n_words, the number of words in the dictionary
    """
    def __init__(self, word2index, index2word, n_words):
        self.word2index = word2index
        self.index2word = index2word
        self.n_words = n_words

    # Use each sentence and instantiate the class properties
    def add_sentence(self, sentence):
        for word in sentence.split(' '): # For each word in the sentence
            if word not in self.word2index: # If word is not seen
                # Add new word
                self.word2index[word] = self.n_words
                self.index2word[self.n_words] = word
                self.n_words += 1

class EncoderRNN(nn.Module):
    """
    The encoder is a GRU in our case.
    It takes the questions matrix as input. For each word in the 
    sentence, it produces a vector and a hidden state; The last one
    will be passed to the decoder in order to initialize it.
    """
    # Initialize encoder
    def __init__(self, input_size, hidden_size): 
        super(EncoderRNN, self).__init__()
        self.hidden_size = hidden_size

        # Embedding layers convert the padded sentences into appropriate vectors
        # The input size is equal to the questions vocabulary
        self.embedding = nn.Embedding(input_size, hidden_size)
        
        # We use a GRU because it's simpler and more efficient (training-wise)
        # than an LSTM
        self.gru = nn.GRU(hidden_size, hidden_size)

    # Forward passes
    def forward(self, input, hidden):
        embedded = self.embedding(input).view(1, 1, -1)
        output = embedded

        # Pass the hidden state and the encoder output to the next word input
        output, hidden = self.gru(output, hidden) 

        return output, hidden

   # PyTorch Forward Passes
    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)

class AttnDecoderRNN(nn.Module):
    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=MAX_LENGTH):
        # Initialize the constructor
        super(AttnDecoderRNN, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        # Combine Fully Connected Layer
        self.attention = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attention_combine = nn.Linear(self.hidden_size * 2,
                                           self.hidden_size)
        # Use dropout
        self.dropout = nn.Dropout(self.dropout_p)

        # Follow with a GRU and a FC layer
        # We use a GRU because it's simpler and more efficient (training-wise)
        # than an LSTM
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, input, hidden, encoder_outputs):
        # Forward passes as from the repo
        embedded = self.embedding(input).view(1, 1, -1)
        embedded = self.dropout(embedded)

        attention_weights = F.softmax(self.attention(torch.cat((embedded[0],
                                                                hidden[0]), 1)),
                                                                 dim=1)
        
        attention_applied = torch.bmm(attention_weights.unsqueeze(0), encoder_outputs.unsqueeze(0))

        output = torch.cat((embedded[0], attention_applied[0]), 1)
        output = self.attention_combine(output).unsqueeze(0)

        # Follow with a ReLU activation function after dropout
        output = F.relu(output)

        # Then, use the GRU
        output, hidden = self.gru(output, hidden)

        # And use softmax as the activation function
        output = F.log_softmax(self.out(output[0]), dim=1)

        return output, hidden, attention_weights

    def init_hidden(self):
        return torch.zeros(1, 1, self.hidden_size)


class Chatbot:
    def __init__(self, max_length = MAX_LENGTH):
        model_path = "model"
        self.encoder = torch.load( model_path +'/encoder.pt', map_location=torch.device('cpu'))
        self.decoder = torch.load(model_path +'/decoder.pt', map_location=torch.device('cpu'))
        self.max_length = max_length
        self.questions = QA_Lang(
            np.load(model_path +'/questions-word2idx.npy', allow_pickle= True).item(),
            np.load(model_path +'/questions-idx2word.npy', allow_pickle= True).item(),
            np.load(model_path +'/questions-n_words.npy', allow_pickle= True).item(),
        )
        self.answers = QA_Lang(
            np.load(model_path +'/answers-word2idx.npy', allow_pickle= True).item(),
            np.load(model_path +'/answers-idx2word.npy', allow_pickle= True).item(),
            np.load(model_path +'/answers-n_words.npy', allow_pickle= True).item(),
        )
        self.name = "Mikabot"
    
    def _preprocess_text(self, sentence):
        sentence = sentence.lower().strip()

        # Convert Unicode string to plain ASCII characters
        normalized_sentence = [c for c in unicodedata.normalize('NFD', sentence) if
                            unicodedata.category(c) != 'Mn']

        # Append the normalized sentence
        sentence = ''
        sentence = ''.join(normalized_sentence)
        
        # Remove punctuation and non-alphabet characters
        sentence = re.sub(r"([.!?])", r" \1", sentence)
        sentence = re.sub(r"[^a-zA-Z.!?]+", r" ", sentence)

        return sentence
    
    def _tensor_from_sentence(self, sentence):
        # For each sentence, get a list of the word indices
        indices = []
        for word in self._preprocess_text(sentence).split(' '):
            if word in self.questions.word2index.keys():
                index = self.questions.word2index[word]
            else:
                index = random.randint(3, self.questions.n_words) 
            indices.append(index)
        indices.append(EOS_TOKEN) # That will help the decoder know when to stop

        # Convert to a PyTorch tensor
        sentence_tensor = torch.tensor(indices, dtype=torch.long).view(-1, 1)

        return sentence_tensor

    def _output_sentence(self, sentence):
        # Get the tensors from the input sentence
        with torch.no_grad():
            input_tensor = self._tensor_from_sentence(sentence)
            input_length = input_tensor.size()[0]
            encoder_hidden = self.encoder.init_hidden()
            encoder_outputs = torch.zeros(self.max_length, self.encoder.hidden_size)

            for ei in range(input_length):
                encoder_output, encoder_hidden = self.encoder(input_tensor[ei],
                                                        encoder_hidden)
                encoder_outputs[ei] += encoder_output[0, 0]

            # Initialize the decoder
            decoder_input = torch.tensor([[SOS_TOKEN]])  # SOS

            decoder_hidden = encoder_hidden

            decoded_words = []
            decoder_attentions = torch.zeros(self.max_length, self.max_length)

            # Combine the decoder output tensors to a sentence
            for di in range(self.max_length):
                decoder_output, decoder_hidden, decoder_attention = self.decoder(
                    decoder_input, decoder_hidden, encoder_outputs)
                decoder_attentions[di] = decoder_attention.data
                topv, topi = decoder_output.data.topk(1)
                if topi.item() == EOS_TOKEN:
                    break
                else:
                    decoded_words.append(self.answers.index2word[topi.item()])

                decoder_input = topi.squeeze().detach()

            return str(' '.join(decoded_words))
    
    def reply(self, sentence):
        '''
        # Takes the input sentence and gives an answer
        - Preprocess the output answers
        - Change bot's name
        - TBA
        '''
        output = self._output_sentence(sentence)
        # Capitalize words
        output = '. '.join(map(lambda s: s.strip().capitalize(), output.split('.')))
        if "Mikabot" in output:
            output.replace("Mikabot", self.name)
        return output

    def test_run(self):
        print(self.reply("where are you?"))
        print(self.reply("hello, how are you?"))
        print(self.reply("that's not funny")) 
        print(self.reply("let's do something fun !"))
        print(self.reply("what's the meaning of life"))
        print(self.reply("I'm hungry can you order pizza"))
        print(self.reply("are you self-aware?"))
        print(self.reply("what do you think about singularity"))
        print(self.reply("why"))
        print(self.reply("humans and robots should work together to make the world a better place. what do you think"))

