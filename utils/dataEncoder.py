import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence


class DataEncoder:
    
    def __init__(self, data,  modelFormat, vocab=None,threshold=None,
                       max_num=None, min_num= None, word_counts=None, 
                 pretrained=None, pretrained_dim=None):
        
        self.data=data
        
        self.modelFormat= modelFormat
        self.max_num = max_num
        self.min_num = min_num
        self.word_counts = word_counts
        self.threshold = threshold
        self.pretrained_dim = pretrained_dim
        

        if pretrained:
          self.pretrained = pretrained
          self.pretrained['<sos>'] = np.random.rand(self.pretrained_dim)
          self.pretrained['<eos>'] = np.random.rand(self.pretrained_dim)
          self.pretrained['<oov>'] = np.random.rand(self.pretrained_dim)
          self.pretrained['<pad>'] = np.random.rand(self.pretrained_dim)

          self.vocab ={word : idx for idx, word in enumerate(list(pretrained.keys()))}
          self.vectors = torch.Tensor(list(pretrained.values()))

        else:
          self.vocab=vocab

        self.idx2wrd = {idx : wrd for wrd, idx in self.vocab.items()}

        if not self.max_num:
            self.max_num = len(self.data)
            
        if not self.min_num:
            self.min_num = 0
            
    def encode(self, test=None, max_len=None):
        
        if self.modelFormat == 'ffnn':
            return self.encode_data_fnn()
        elif self.modelFormat == 'rnn':
            return self.encode_data_rnn(test, max_len)
        elif self.modelFormat == 'svm':
            return self.encode_data_svm(test)
        else:
            raise('I make sure you use a compatible model bringus')
    
    def encode_data_svm(self, test=None):

        encoded_data = []
        empty_vec = np.zeros(len(self.vocab))
        
        data = self.data
        
        for line in data:

            encoded_line = np.copy(empty_vec)

            for token in line:

                if self.threshold:

                    if self.word_counts[token] > self.threshold:
                        encoded_line[self.vocab[token]] += 1

                    else:
                        continue

                encoded_line[self.vocab[token]] += 1

            encoded_data.append(encoded_line)

        return np.array(encoded_data)
        
    
    def encode_data_fnn(self):
    
        encoded_data = []
        empty_vec = np.zeros(len(self.vocab))

        for line in self.data:

            encoded_line = np.copy(empty_vec)
            
            if type(line) == str:
                line = line.split(' ')
                
            for token in line:

                if self.threshold:

                    if word_counts[token] > self.min_num and word_counts[token] < self.max_num:
                        encoded_line[vocab[token]] += 1

                    else:
                        continue

                encoded_line[self.vocab[token]] += 1

            encoded_data.append(encoded_line)

        return np.array(encoded_data)
    
    def encode_data_rnn(self, test=None, max_len=None, add_start_end=None):
        
        encoded_data = []
        
        if test:
            data=test
        
        else:
            data=self.data
        
        for line in data:

            encoded_line = []
            
            if type(line) == str:
                line = line.split(' ')

            if add_start_end:
              line.insert(0, '<sos>')
              line.append('<eos>')

            for token in line:

                if self.threshold:
                    
                    if token not in self.vocab:
                        encoded_line.append(vocab['oov'])
                        
                    else:
                        if word_counts[token] > self.min_num and word_counts[token] < self.max_num:
                            encoded_line.append(self.vocab[token])

                        else:
                            continue

                if token not in self.vocab:
                  encoded_line.append(self.vocab['<oov>'])

                else:
                  encoded_line.append(self.vocab[token])
            
            if max_len:
                encoded_data.append(torch.LongTensor(encoded_line[:max_len]))
            else:
                encoded_data.append(torch.LongTensor(encoded_line))


        if self.pretrained:
          return pad_sequence(encoded_data, batch_first=True, padding_value=self.vocab['<pad>'])

        else:
          return pad_sequence(encoded_data, batch_first=True, padding_value=0)
    
    def decode(self, encoded_line):
        
        not_decodes = [self.vocab['<oov>'], self.vocab['<eos>'], 
                      self.vocab['<sos>'], self.vocab['<pad>']]

        return [self.idx2wrd[i.item()] for i in encoded_line if i.item() not in not_decodes]
    
    def encodeMultiHot(line, vocab, test=None):
    

        empty_vec = np.zeros(len(vocab))

        for token in line:

            if token in vocab:
                empty_vec[vocab[token]] += 1


        return np.array(empty_vec)