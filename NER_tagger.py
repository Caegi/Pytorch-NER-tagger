"""
CARDOSO_NECHAB_M2 project : NER tagging

This project aims to implement a NER Tagger with Pytorch. We will be using the English CONLL 2003 data set.

Data download & description

--------
"""

import matplotlib.pyplot as plt
from transformers import AutoTokenizer
from collections import Counter
import torch
import torch.nn as nn
from random import shuffle

from urllib.request import urlretrieve
urlretrieve('https://raw.githubusercontent.com/pranabsarkar/Conll_task/master/conll-2003/eng.train','eng.train')
urlretrieve('https://raw.githubusercontent.com/pranabsarkar/Conll_task/master/conll-2003/eng.testa','eng.testa')

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

"""
The CONLL 2003 dataset encodes each token on a single line followed by its annotation. A token line is a quadruple:

> (token,tag,chunk,named entity)

A named entity tagger aims to predict the named entity annotations given the raw tokens. The NER tags follows the IOB convention.

* **I** stands for **Inside** and is used to flag tokens that are part of a named entity.

* **B** stands for **Begin** and is used to flag a token starting a new entity when the preceding token is already part of an entity.

* **O** stands for **Outside** and is used to flag tokens that are not part of a named entity.


The I and B Tag are followed by a specifier. For instance I-PER means that the named entity refers to a person, I-ORG means that the entity is refers to an Organisation.


Sentences are separated by a blank line. The train file is `eng.train`, the dev file is `eng.testa`.
"""

def vocabulary(filename,input_vocab,postag_vocab=False,padding='<pad>',unknown='<unk>'):
    #input_vocab is a boolean flag that tells if we extract input or output vocabulary
    #postag_vocab is a boolean flag that tells if we extract pad tokens or not
    #the two optional flags indicate that a padding and an unknown token
    #have to be added to the vocabulary if their value is not None
    set_vocab = set()

    idx2sym = {}
    sym2idx = {}

    #char_vocab is a boolean flag that tells if we extract char symbols or language codes
    istream = open('eng.train')
    for _, line in enumerate(istream):
      if line and not line.isspace():

        if not(postag_vocab):
          token = line.strip().split(" ")[0]

          if input_vocab:
            set_vocab.add(token)

          else:
            ner_tag = line.strip().split(" ")[-1]
            set_vocab.add(ner_tag)

        else:
          pos_tag = line.strip().split(" ")[1]
          set_vocab.add(pos_tag)

    for i,token in enumerate(set_vocab):
      sym2idx[token] = i
      idx2sym[i] = token

    # padding
    if padding is not None:
      sym2idx[padding] = len(sym2idx)
      idx2sym[len(idx2sym)] = padding

    # unknown words
    if unknown is not None:
      sym2idx[unknown] = len(sym2idx)
      idx2sym[len(idx2sym)] = unknown

    return idx2sym, sym2idx

"""
Now we implement three functions:

* One that performs padding
* The second will encode a sequence of tokens (or a sequence of tags) on integers
* The third will decode as sequence of symbols from integers to strings

"""

def pad_sequence(sequence,pad_size,pad_token):
    #returns a list with additional pad tokens if needed
    if len(sequence) < pad_size:
      sequence.extend( [pad_token for _ in range(pad_size - len(sequence))] )
    return sequence

# unk_token needs to be specified
def code_sequence(sequence,coding_map,unk_token=None):
    #takes a list of strings and returns a list of integers
    return [coding_map[c] if c in coding_map else coding_map[unk_token] for c in sequence]

def decode_sequence(sequence,decoding_map):
    #takes a list of integers and returns a list of strings
    return [decoding_map[idx] for idx in sequence]

# test pad_sequence
print(f'list["a", "b"], pad_size = 2: {pad_sequence(["a", "b"], 2, "<pad>")}')
print(f'list["a", "b", "c"], pad_size = 2: {pad_sequence(["a", "b", "c"], 2, "<pad>")}')
print(f'list["a", "b", "c"], pad_size = 4: {pad_sequence(["a", "b", "c"], 4, "<pad>")}')
print(f'list["a", "b", "c"], pad_size = 99: length is {len(pad_sequence(["a", "b", "c"], 99, "<pad>"))} - {pad_sequence(["a", "b", "c"], 99, "<pad>")}')

# test code sequence
symbs = ["I","want","to","sleep"]
symbs_last_unk = ["I","skgflksljdgkjpsdjpfkskopdf","want","to","famkdflmkadfkml","sleep","hsasdkjaldnfklsnclvkn"]
coding_map = {"I":0, "want":1, "to":2, "sleep":3, "<unk>":4}

print(f"symbs: {symbs}")
print(f"coding_map(symbols to int): {coding_map}")
print(f"result of code_sequence(symbs, coding_map): {code_sequence(symbs, coding_map)}\n")

print(f"symbs_last_unk: {symbs_last_unk}")
print(f"result of code_sequence(symbs_last_unk, coding_map, unk_token='<unk>'): {code_sequence(symbs_last_unk, coding_map, unk_token='<unk>')}")

decoding_map = {0:"I",1:"want",2:"to",3:"sleep"}
inds = [0,1,2,3]

print(f"inds: {inds}")
print(f"decoding_map(int to symbols): {decoding_map} \n")
print(f"result of decode_sequence(inds,decoding_map): {decode_sequence(inds,decoding_map)}")

"""
This is a class in charge of generating randomized batches of data from the dataset. We start by implementing tree functions for reading the textfile
"""

def read_conll_tokens(conllfilename):
    """
    Reads a CONLL 2003 file and returns a list of sentences.
    A sentence is a list of strings (tokens)
    """
    sentences = []
    sentence = []

    istream = open(conllfilename)
    for _, line in enumerate(istream):

      if line and line.isspace():
        sentences.append(sentence)
        sentence = []

      elif line and not line.isspace():
        sentence.append(line.split(" ")[0])

    return sentences


def read_conll_ner_tags(conllfilename):
    """
    Reads a CONLL 2003 file and returns a list of sentences.
    A sentence is a list of strings (NER-tags)
    """
    sentences = []
    sentence_tags = []

    istream = open(conllfilename)
    for _, line in enumerate(istream):

      if line and line.isspace():
        sentences.append(sentence_tags)
        sentence_tags = []

      elif line and not line.isspace():
        sentence_tags.append(line.split(" ")[-1].strip())

    return sentences


# function used to get the pos tags to add pos tags embeddings to the model
def read_conll_pos_tags(conllfilename):
    """
    Reads a CONLL 2003 file and returns a list of sentences.
    A sentence is a list of strings (POS-tags)
    """
    sentences = []
    sentence_tags = []

    istream = open(conllfilename)
    for _, line in enumerate(istream):

      if line and line.isspace():
        sentences.append(sentence_tags)
        sentence_tags = []

      elif line and not line.isspace():
        sentence_tags.append(line.split(" ")[1])

    return sentences

"""
text used for testing: 

-DOCSTART- -X- -X- O <br>


EU NNP I-NP I-ORG <br>

rejects VBZ I-VP O<br>

German JJ I-NP I-MISC<br>

call NN I-NP O<br>

to TO I-VP O<br>

boycott VB I-VP O<br>

British JJ I-NP I-MISC<br>

lamb NN I-NP O<br>

. . O O<br>

Testing read_conll_tokens
"""

print(f"Expected 1st sentence: ['-DOCSTART-']")
print(f"1st sentence: {read_conll_tokens('eng.train')[0]}")

print(f"Expected 2nd sentence: ['EU', 'rejects', 'German', 'call', 'to', 'boycott', 'British', 'lamb', '.']")
print(f"2nd sentence: {read_conll_tokens('eng.train')[1]}")

"""Testing read_conll_ner_tags"""

print(f"Expected 1st sentence: ['O']")
print(f"1st sentence: {read_conll_ner_tags('eng.train')[0]}")

print(f"Expected 2nd sentence: ['I-ORG', 'O', 'I-MISC', 'O', 'O', 'O', 'I-MISC', 'O', 'O']")
print(f"2nd sentence: {read_conll_ner_tags('eng.train')[1]}")

"""Testing read_conll_pos_tags"""

print(f"Expected 1st sentence: ['-X-']")
print(f"1st sentence: {read_conll_pos_tags('eng.train')[0]}")

print(f"Expected 2nd sentence: ['NNP', 'VBZ', 'JJ', 'NN', 'TO', 'VB', 'JJ', 'NN', '.']")
print(f"2nd sentence: {read_conll_pos_tags('eng.train')[1]}")

"""
Now we implement the class.
"""

class DataGenerator:

        #Reuse all relevant helper functions defined above to solve the problems
        def __init__(self,conllfilename, parentgenerator = None, pad_token='<pad>',unk_token='<unk>'):

              if parentgenerator is not None: #Reuse the encodings of the parent if specified

                  self.pad_token      = parentgenerator.pad_token
                  self.unk_token      = parentgenerator.unk_token
                  self.input_sym2idx  = parentgenerator.input_sym2idx
                  self.input_idx2sym  = parentgenerator.input_idx2sym
                  self.output_sym2idx = parentgenerator.output_sym2idx
                  self.output_idx2sym = parentgenerator.output_idx2sym
                  self.pos_idx2sym    = parentgenerator.pos_idx2sym
                  self.pos_sym2idx    = parentgenerator.pos_sym2idx

              else:                           #Creates new encodings
                  self.pad_token = pad_token
                  self.unk_token = unk_token


                  self.input_idx2sym,self.input_sym2idx   = vocabulary(conllfilename, True)
                  self.output_idx2sym,self.output_sym2idx = vocabulary(conllfilename, False)
                  self.pos_idx2sym,self.pos_sym2idx = vocabulary(conllfilename, False, postag_vocab=True)


              self.Xtokens = read_conll_tokens(conllfilename)
              self.Ytokens = read_conll_ner_tags(conllfilename)
              self.Postag_tokens = read_conll_pos_tags(conllfilename)

        def generate_batches(self,batch_size):

              #This is an example generator function yielding one batch after another
              #Batches are lists of lists

              assert(len(self.Xtokens) == len(self.Ytokens) == len(self.Postag_tokens))

              N     = len(self.Xtokens)
              idxes = list(range(N))

              #Data ordering (try to explain why these 2 lines make sense...)
              # shuffling is a good idea to stop the model from learning the order of the data while training
              # we do not want the model to "overfit" on the order of the data, if it does it will not generalise well
              shuffle(idxes)
              # this line allows for efficient tensor computation since all sequences of a given size will be computed close to one another
              # this allows to not have to add too many pad tokens
              idxes.sort(key=lambda idx: len(self.Xtokens[idx]))

              #batch generation
              bstart = 0

              while bstart < N:
                 bend        = min(bstart+batch_size,N)
                 batch_idxes = idxes[bstart:bend]
                 batch_len   = max(len(self.Xtokens[idx]) for idx in batch_idxes)

                 seqX = [ pad_sequence(self.Xtokens[idx],batch_len,self.pad_token) for idx in batch_idxes]
                 seqY = [ pad_sequence(self.Ytokens[idx],batch_len,self.pad_token) for idx in batch_idxes]
                 seqP = [ pad_sequence(self.Postag_tokens[idx],batch_len,self.pad_token) for idx in batch_idxes]
                 seqX = [ code_sequence(seq,self.input_sym2idx,self.unk_token) for seq in seqX]
                 seqY = [ code_sequence(seq,self.output_sym2idx) for seq in seqY]
                 seqP = [ code_sequence(seq,self.pos_sym2idx) for seq in seqP]

                 assert(len(seqX) == len(seqY) == len(seqP))
                 yield (seqX,seqY,seqP)

                 bstart += batch_size

trainset = DataGenerator('eng.train')
validset = DataGenerator('eng.testa',parentgenerator = trainset)

import torch.optim as optim


class NERtagger(nn.Module):

      def __init__(self,traingenerator, embedding_size,hidden_size,print_tensor_shapes_forward=False,device='cpu'):

        super(NERtagger, self).__init__()
        self.embedding_size    = embedding_size
        self.hidden_size       = hidden_size
        self.pad_index = traingenerator.output_sym2idx[traingenerator.pad_token]
        self.allocate_params(traingenerator,device)
        self.print_tensor_shapes_forward = print_tensor_shapes_forward

      def load(self,filename):
        self.load_state_dict(torch.load(filename))

      def allocate_params(self,datagenerator,device): #create fields for nn Layers
        self.token_embs = torch.nn.Embedding(len(datagenerator.input_sym2idx), self.embedding_size, padding_idx=self.pad_index, device=device)
        self.lstm = torch.nn.LSTM(self.embedding_size, self.hidden_size, device=device)
        self.output = torch.nn.Linear(self.hidden_size, len(datagenerator.output_sym2idx), device=device)

      def forward(self, Xinput):
        """
        Xinput: [batch_size, seq_length] - Input tokens
        """
        xemb = self.token_embs(Xinput) # shape: [batch_size, seq_length, embedding_size]
        if self.print_tensor_shapes_forward: print(f"expected shape [batch_size, seq_length, embedding_size]: {xemb.shape}")

        lstm_output, _ = self.lstm(xemb) # shape: [batch_size, seq_length, hidden_size]
        if self.print_tensor_shapes_forward: print(f"expected shape [batch_size, seq_length, hidden_size]: {lstm_output.shape}")

        logits = self.output(lstm_output) # shape: [batch_size, seq_length, output_vocab_size(number gold classes)]
        if self.print_tensor_shapes_forward: print(f"expected shape [batch_size, seq_length, output_vocab_size]: {logits.shape}")

        return nn.LogSoftmax(dim=-1)(logits) # shape: [batch_size, seq_length, output_vocab_size(number gold classes)]

      # this function trains the model, plot the accuracy and loss over valid and train and return dev accuracy(the objective to maximise while running random hyperparameter search)
      def train_model(self,traingenerator,validgenerator,epochs,batch_size,patience,verbose=True,device='cpu',learning_rate=0.001):
        self.minloss = 10000000 #the min loss found so far on validation data
        patience_count = 0 #the number of times dev loss has increased compared to last epoch dev loss

        i_epoch = 0
        optimizer = optim.Adam(self.parameters(),lr=learning_rate)
        loss_fnc = nn.CrossEntropyLoss()
        self.to(device)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        while i_epoch < epochs:
          if verbose: print(f"\nEpoch {i_epoch} out of {epochs}")
          self.train() # Tells PyTorch that we are in training mode
          batch_accurracies = []
          batch_losses = []
          batch_sizes = []

          for (seqX,seqY,seqP) in traingenerator.generate_batches(batch_size):
            X = torch.LongTensor(seqX).to(device)
            Y = torch.LongTensor(seqY).to(device)

            optimizer.zero_grad()
            Ypred = self.forward(X)

            batch_size,seq_len = Y.shape
            Ypred = Ypred.view(batch_size*seq_len,-1)
            Y    = Y.view(batch_size*seq_len)
            loss_training = loss_fnc(Ypred,Y)
            loss_training.backward()
            optimizer.step()

            with torch.no_grad():
              batch_losses.append(loss_training.item())

              #Accurracy computation
              mask    = (Y != self.pad_index)
              Yargmax = torch.argmax(Ypred,dim=1)
              correct = torch.sum((Yargmax == Y) * mask)
              total   = torch.sum(mask)
              batch_accurracies.append(float(correct)/float(total))

          L = len(batch_losses)
          epoch_mean_train_accuracy = sum(batch_accurracies)/L
          epoch_mean_train_loss = sum(batch_losses)/L

          if verbose: print(f"[train]  mean_loss = {epoch_mean_train_loss}, mean_acc = {epoch_mean_train_accuracy}")
          train_losses.append(epoch_mean_train_loss)
          train_accuracies.append(epoch_mean_train_accuracy)

          best_loss = self.minloss
          epoch_mean_val_loss, epoch_mean_val_accuracy = self.validate(validgenerator,batch_size,verbose=verbose,device=device,save_min_model=True)
          val_accuracies.append(epoch_mean_val_accuracy)
          val_losses.append(epoch_mean_val_loss)

          # if current loss is higher or equal to min loss (see validate)
          if self.minloss == best_loss:
            patience_count += 1

          else:
            patience_count = 0

          if patience_count == patience:
            if verbose: print(f"Stopped early at epoch {i_epoch+1}")
            break

          i_epoch += 1

        epoch_list = list(range(1, len(train_losses) + 1))
        plt.figure(figsize=(12, 5))

        # Plot loss over epochs
        plt.subplot(1, 2, 1)
        plt.plot(epoch_list, train_losses, label='Training Loss')
        plt.plot(epoch_list, val_losses, label='Dev Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Plot accuracy over epochs
        plt.subplot(1, 2, 2)
        plt.plot(epoch_list, train_accuracies, label='Train Accuracy')
        plt.plot(epoch_list, val_accuracies, label='Dev Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()
        plt.show()

        last_epoch_val_acc = epoch_mean_val_accuracy
        return last_epoch_val_acc

      def validate(self,datagenerator,batch_size,verbose=True,device='cpu',save_min_model=False):
          batch_accurracies = []
          batch_losses      = []

          device = torch.device(device)
          pad_index = datagenerator.output_sym2idx[datagenerator.pad_token]
          loss_fnc  = nn.CrossEntropyLoss(ignore_index=pad_index)

          for (seqX,seqY,seqP) in datagenerator.generate_batches(batch_size):

                with torch.no_grad():
                  X = torch.LongTensor(seqX).to(device)
                  Y = torch.LongTensor(seqY).to(device)

                  Yhat = self.forward(X)

                  #Flattening and loss computation
                  batch_size,seq_len = Y.shape
                  Yhat = Yhat.view(batch_size*seq_len,-1)
                  Y    = Y.view(batch_size*seq_len)
                  loss = loss_fnc(Yhat,Y)
                  batch_losses.append(loss.item())

                  #Accurracy computation
                  mask    = (Y != pad_index)
                  Yargmax = torch.argmax(Yhat,dim=1)
                  correct = torch.sum((Yargmax == Y) * mask)
                  total   = torch.sum(mask)
                  batch_accurracies.append(float(correct)/float(total))

          L = len(batch_losses)
          valid_loss = sum(batch_losses)/L

          if save_min_model and valid_loss < self.minloss:
            self.minloss = valid_loss
            torch.save(self.state_dict(), 'tagger_params.pt')


          valid_accuracy = sum(batch_accurracies)/L
          if verbose: print('[valid] mean Loss = %f | mean accurracy = %f'%(valid_loss,valid_accuracy))
          return valid_loss, valid_accuracy

"""
Testing forward
"""

simple_tagger = NERtagger(trainset, embedding_size=64, hidden_size=128, print_tensor_shapes_forward=True, device=device)

batch_size = 10

for (seqX, seqY, seqP) in trainset.generate_batches(batch_size=batch_size):
  print(f"batch_size = 10")
  print(f"seq_length = {len(seqX[0])}")
  print(f"embedding_size = {simple_tagger.embedding_size}")
  print(f"hidden_size = {simple_tagger.hidden_size}")
  print(f"output_vocab_size = {len(trainset.output_idx2sym)}\n")

  print(f"expected shape [batch_size, seq_length, output_vocab_size]: {simple_tagger(torch.tensor(seqX).to(device)).shape}")
  break

"""
The main program is the following. 
"""

#NERtagger with multihead attention and postag embeddings
class NERtagger_improved(NERtagger):

    def __init__(self,traingenerator,embedding_size,hidden_size,nb_attn_heads,print_tensor_shapes_forward=False,device='cpu'):
        self.nb_attn_heads = nb_attn_heads
        super().__init__(traingenerator, embedding_size, hidden_size, print_tensor_shapes_forward=print_tensor_shapes_forward, device=device)

    # Overriding
    def allocate_params(self,datagenerator,device): #create fields for nn Layers
        self.token_embs = torch.nn.Embedding(len(datagenerator.input_sym2idx), self.embedding_size, padding_idx=self.pad_index, device=device)
        self.postag_embs = torch.nn.Embedding(len(datagenerator.pos_idx2sym), self.embedding_size, padding_idx=self.pad_index, device=device)
        self.lstm = torch.nn.LSTM(self.embedding_size*2, self.hidden_size, device=device) # self.embedding_size*2: since the input is the 2 embeddings: token_embs and postag_embs stacked
        self.multihead_attn = nn.MultiheadAttention(embed_dim=self.hidden_size, num_heads=self.nb_attn_heads, dropout=0.1)
        self.output = torch.nn.Linear(self.hidden_size, len(datagenerator.output_sym2idx), device=device)

    # Overriding
    def forward(self, Xinput, Xpostag):
        """
        Xinput: [batch_size, seq_length] - Input tokens
        Xpostag: [batch_size, seq_length] - POS tags
        """
        xemb = self.token_embs(Xinput) # shape: [batch_size, seq_length, embedding_size]
        xemb_pos_tag = self.postag_embs(Xpostag) # shape: [batch_size, seq_length, embedding_size]
        combined_emb = torch.cat((xemb, xemb_pos_tag), dim=-1) # shape: [batch_size, seq_length, embedding_size * 2]
        lstm_output, _ = self.lstm(combined_emb) # shape: [seq_length, batch_size, hidden_size]
        attn_output, _ = self.multihead_attn(lstm_output, lstm_output, lstm_output) # shape: [seq_length, batch_size, hidden_size]
        # element wise addition to combine the attention and the lstm output
        logits = self.output(lstm_output + attn_output) # shape:  # [seq_length, batch_size, num_classes]
        return nn.LogSoftmax(dim=-1)(logits)  # [seq_length, batch_size, num_classes]

    # Overridding
    def train_model(self,traingenerator,validgenerator,epochs,batch_size,patience,verbose=True,device='cpu',learning_rate=0.001):
        self.minloss = 10000000 #the min loss found so far on validation data
        patience_count = 0 #the number of times dev loss has increased compared to last epoch dev loss
        i_epoch = 0
        optimizer = optim.Adam(self.parameters(),lr=learning_rate)
        loss_fnc = nn.CrossEntropyLoss()
        self.to(device)

        train_losses = []
        train_accuracies = []
        val_losses = []
        val_accuracies = []

        while i_epoch < epochs:
          if verbose: print(f"\nEpoch {i_epoch} out of {epochs}")
          self.train() # Tells PyTorch that we are in training mode
          batch_accurracies = []
          batch_losses = []
          batch_sizes = []

          for (seqX,seqY,seqP) in traingenerator.generate_batches(batch_size):
            X = torch.LongTensor(seqX).to(device)
            Y = torch.LongTensor(seqY).to(device)
            P = torch.LongTensor(seqP).to(device)

            optimizer.zero_grad()
            Ypred = self.forward(X,P)
            batch_size,seq_len = Y.shape
            Ypred = Ypred.view(batch_size*seq_len,-1)
            Y    = Y.view(batch_size*seq_len)
            loss_training = loss_fnc(Ypred,Y)
            loss_training.backward()
            optimizer.step()

            with torch.no_grad():
              batch_losses.append(loss_training.item())

              #Accurracy computation
              mask    = (Y != self.pad_index)
              Yargmax = torch.argmax(Ypred,dim=1)
              correct = torch.sum((Yargmax == Y) * mask)
              total   = torch.sum(mask)
              batch_accurracies.append(float(correct)/float(total))

          L = len(batch_losses)
          epoch_mean_train_accuracy = sum(batch_accurracies)/L
          epoch_mean_train_loss = sum(batch_losses)/L

          if verbose: print(f"[train]  mean_loss = {epoch_mean_train_loss}, mean_acc = {epoch_mean_train_accuracy}")

          train_losses.append(epoch_mean_train_loss)
          train_accuracies.append(epoch_mean_train_accuracy)

          best_loss = self.minloss
          epoch_mean_val_loss, epoch_mean_val_accuracy = self.validate(validgenerator,batch_size,verbose=verbose,device=device,save_min_model=True)
          val_accuracies.append(epoch_mean_val_accuracy)
          val_losses.append(epoch_mean_val_loss)

          # if current loss is higher or equal to min loss (see validate)
          if self.minloss == best_loss:
            patience_count += 1

          else:
            patience_count = 0

          if patience_count == patience:
            print(f"Stopped early at epoch {i_epoch+1}")
            break

          i_epoch += 1

        # plotting
        epoch_list = list(range(1, len(train_losses) + 1))
        plt.figure(figsize=(12, 5))

        # Plot loss over epochs
        plt.subplot(1, 2, 1)
        plt.plot(epoch_list, train_losses, label='Training Loss')
        plt.plot(epoch_list, val_losses, label='Dev Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Loss over Epochs')
        plt.legend()

        # Plot accuracy over epochs
        plt.subplot(1, 2, 2)
        plt.plot(epoch_list, train_accuracies, label='Train Accuracy')
        plt.plot(epoch_list, val_accuracies, label='Dev Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Accuracy over Epochs')
        plt.legend()

        plt.show()

        last_epoch_val_acc = epoch_mean_val_accuracy
        return last_epoch_val_acc

    # Override
    def validate(self,datagenerator,batch_size,verbose=True,device='cpu',save_min_model=False):
        batch_accurracies = []
        batch_losses      = []

        device = torch.device(device)
        pad_index = datagenerator.output_sym2idx[datagenerator.pad_token]
        loss_fnc  = nn.CrossEntropyLoss(ignore_index=pad_index)

        for (seqX,seqY,seqP) in datagenerator.generate_batches(batch_size):

            with torch.no_grad():
              X = torch.LongTensor(seqX).to(device)
              Y = torch.LongTensor(seqY).to(device)
              P = torch.LongTensor(seqP).to(device)

              Yhat = self.forward(X,P)

              #Flattening and loss computation
              batch_size,seq_len = Y.shape
              Yhat = Yhat.view(batch_size*seq_len,-1)
              Y    = Y.view(batch_size*seq_len)
              loss = loss_fnc(Yhat,Y)
              batch_losses.append(loss.item())

              #Accurracy computation
              mask    = (Y != pad_index)
              Yargmax = torch.argmax(Yhat,dim=1)
              correct = torch.sum((Yargmax == Y) * mask)
              total   = torch.sum(mask)
              batch_accurracies.append(float(correct)/float(total))

        L = len(batch_losses)
        valid_loss = sum(batch_losses)/L

        if save_min_model and valid_loss < self.minloss:
            self.minloss = valid_loss
            torch.save(self.state_dict(), 'tagger_params.pt')

        valid_accuracy = sum(batch_accurracies)/L
        if verbose: print('[valid] mean Loss = %f | mean accurracy = %f'%(valid_loss,valid_accuracy))
        return valid_loss, valid_accuracy

"""
Results before managing unknown tokens
Training
"""

simple_tagger   = NERtagger(trainset, embedding_size=64, hidden_size=128, device=device)
simple_tagger.train_model(trainset, validset, epochs=20, batch_size=32, patience=3, device=device)

improved_tagger   = NERtagger_improved(trainset, embedding_size=64, hidden_size=128, nb_attn_heads=4)
improved_tagger.train_model(trainset, validset, epochs=20, batch_size=32, patience=3, device=device)

"""
Adding multihead attention and pos tag embeddings allows the model to reach a good accuracy faster, so it learns faster.
"""

"""
Results while managing unknown tokens
Preprocessing changed for unknown token management
"""

subword_tokenizer = AutoTokenizer.from_pretrained("bert-base-multilingual-cased")

def vocabulary(filename,input_vocab,postag_vocab=False,padding='<pad>',unknown='<unk>',subwords=False):
    #input_vocab is a boolean flag that tells if we extract input or output vocabulary
    #postag_vocab is a boolean flag that tells if we extract pad tokens or not
    #the two optional flags indicate that a padding and an unknown token
    #have to be added to the vocabulary if their value is not None

    set_vocab = set()
    idx2sym = {}
    sym2idx = {}
    freq = Counter()

    #char_vocab is a boolean flag that tells if we extract char symbols or language codes
    istream = open('eng.train')

    for line in istream:
      if line and not line.isspace():

        if not(postag_vocab):
          token = line.strip().split(" ")[0]

          if input_vocab:

            if subwords: subtokens = subword_tokenizer.tokenize(token)
            set_vocab.update(subtokens if subwords else token)
            freq.update(subtokens if subwords else token)

          else:
            ner_tag = line.strip().split(" ")[-1]
            set_vocab.add(ner_tag)

        else:
          pos_tag = line.strip().split(" ")[1]
          set_vocab.add(pos_tag)

    for i,token in enumerate(set_vocab):
      sym2idx[token] = i
      idx2sym[i] = token

    # padding
    if padding is not None:
      sym2idx[padding] = len(sym2idx)
      idx2sym[len(idx2sym)] = padding

    # unknown words
    if unknown is None: unknown = min(freq, key=freq.get)

    sym2idx[unknown] = len(sym2idx)
    idx2sym[len(idx2sym)] = unknown

    return idx2sym, sym2idx

def read_conll_tokens(conllfilename, subwords=False):
    """
    Reads a CONLL 2003 file and returns a list of sentences.
    A sentence is a list of strings (tokens)
    """
    sentences = []
    sentence = []

    istream = open(conllfilename)
    for _, line in enumerate(istream):

      if line and line.isspace():
        sentences.append(sentence)
        sentence = []

      elif line and not line.isspace():
        token = line.split(" ")[0]
        if subwords: subtokens = subword_tokenizer.tokenize(token)
        sentence.extend(subtokens if subwords else [token])

    return sentences

def read_conll_ner_tags(conllfilename, subwords=False):
    """
    Reads a CONLL 2003 file and returns a list of sentences.
    A sentence is a list of strings (NER-tags)
    """
    sentences = []
    sentence_tags = []

    istream = open(conllfilename)

    for _, line in enumerate(istream):

      if line and line.isspace():
        sentences.append(sentence_tags)
        sentence_tags = []

      elif line and not line.isspace():

        if subwords: subtokens = subword_tokenizer.tokenize(line.split(" ")[0])
        tag = line.split(" ")[-1].strip()
        sentence_tags.extend([tag]*len(subtokens) if subwords else [tag])

    return sentences

# function used to get the pos tags to add pos tags embeddings to the model
def read_conll_pos_tags(conllfilename, subwords=False):
    """
    Reads a CONLL 2003 file and returns a list of sentences.
    A sentence is a list of strings (POS-tags)
    """
    sentences = []
    sentence_tags = []

    istream = open(conllfilename)

    for _, line in enumerate(istream):

      if line and line.isspace():
        sentences.append(sentence_tags)
        sentence_tags = []

      elif line and not line.isspace():

        if subwords: subtokens = subword_tokenizer.tokenize(line.split(" ")[0])
        tag = line.split(" ")[1]
        sentence_tags.extend([tag]*len(subtokens) if subwords else [tag])

    return sentences

trainset = DataGenerator('eng.train')
validset = DataGenerator('eng.testa',parentgenerator = trainset)

"""Training with unknown token management"""

simple_tagger   = NERtagger(trainset, embedding_size=64, hidden_size=128, device=device)
simple_tagger.train_model(trainset, validset, epochs=20, batch_size=32, patience=3, verbose=False, device=device) #use verbose=True to see results of the validate function

improved_tagger   = NERtagger_improved(trainset, embedding_size=64, hidden_size=128, nb_attn_heads=4)
improved_tagger.train_model(trainset, validset, epochs=20, batch_size=32, patience=3, verbose=False, device=device) #use verbose=True to see results of the validate function

"""
The results are worse with the way we managed the unknown tokens.
Adding the multihead attention and pos tag embeddings allows the model to reach a better accuracy.
"""

"""
Hyperparameter Random Grid Search (the model used was the one made before managing unknown tokens)
"""

import optuna

# the objective of the trial: this initalizes the embedding size, the hidden size and the number of epochs
# It runs an NER tagger, and returns the dev accuracy of the last epoch
def objective(trial):
    embedding_size = trial.suggest_int("embedding_size", 32, 512, step=32)
    hidden_size = trial.suggest_int("hidden_size", 32, 512, step=32)
    nb_attn_heads = trial.suggest_int("nb_attn_heads", 2, 4, step=2)

    NER_tagger = NERtagger_improved(
                trainset,
                embedding_size=embedding_size,
                hidden_size=hidden_size,
                nb_attn_heads=nb_attn_heads,
                device=device
    )

    dev_accuracy = NER_tagger.train_model(trainset, validset, epochs=15, batch_size=32, patience=3, verbose=False, device=device)
    return dev_accuracy

# use this function to run random search of hyperparameters, (takes a long time to run)
def search_hyperparameters(n_trials):
  # defining the random search study that will maximize the objective of the trial (val accuracy)
  # study: Optuna Study object
  study = optuna.create_study(direction="maximize", sampler=optuna.samplers.RandomSampler())

  # trying n_trials combination of random hyper-parameters
  study.optimize(objective, n_trials=n_trials)

  return study

study = search_hyperparameters(n_trials=40)

"""
These were the best hyperparameters found

Best Trial: [I 2024-11-22 15:01:11,728] Trial 36 finished with value: 0.9394577506314833 and parameters: {'embedding_size': 480, 'hidden_size': 128, 'nb_attn_heads': 4}. Best is trial 36 with value: 0.9394577506314833.


Last Trial: [I 2024-11-22 15:26:49,968] Trial 39 finished with value: 0.9187277174029199 and parameters: {'embedding_size': 288, 'hidden_size': 512, 'nb_attn_heads': 4}. Best is trial 36 with value: 0.9394577506314833.
"""

print(optuna.visualization.plot_optimization_history(study))

print(optuna.visualization.plot_parallel_coordinate(study))

"""
In this graph we can see that, overall, the combination of around 150 embedding size and around 200 hidden_size worked well. The combination of embedding_size of around 450 and of hidden size of around 150 worked well too. The combination of a low embedding_size with a high hidden_size overall did not work very well.

We can deduce that having a high embedding size is important, and that we should avoid setting a small embedding size with a big hidden size. Moreover, choosing between 2 or 4 attention heads did not improve the model significantly.
"""

print(optuna.visualization.plot_param_importances(study))

"""
We can confirm that the embedding size was the most important hyperparameter to tune, and the number of attention heads the least important hyperparameter to tune
"""