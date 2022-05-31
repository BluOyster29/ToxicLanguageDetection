import re
from tqdm.auto import tqdm
import numpy as np 
from torch.utils.data import Dataset, DataLoader
import torch 

bad_strings = re.compile(r'[\n"\.,]')
encodings = re.compile(u'\\?[a-z]+[0-9]·')

def updateVec(row):
    
    vec = []
    
    vec.append(row.toxic)
    vec.append(row.severe_toxic)
    vec.append(row.obscene)
    vec.append(row.threat)
    vec.append(row.insult)
    vec.append(row.identity_hate)
    
    return vec

def preprocessString(input_string):
    
    string = re.sub(bad_strings, '' , input_string)
    string = string.lower()
    string = re.sub(encodings, '', string)
    
    return string

def processRawDataFromCSV(input_csv, test=None):
    
    x = [] 
    y = []
    
    for i in tqdm(range(len(input_csv))):
        row = input_csv.loc[i]
        x.append(preprocessString(row.comment_text))
        
        if not test:
            y.append(updateVec(row))
        
    return x, y

def processRawDataFromList(x):
    
    return [preprocessString(i) for i in tqdm(x)]

def buildVocab(training_data, tokenised=None, rnn=None):
    
    vocab={}
    word_counts = {}
    
    vocab['<sos>'] = 0
    vocab['<eos>'] = 1
    vocab['<oov>'] = 2
    
    for line in training_data:
        
        
        for token in line:
            
            if token not in vocab:
                vocab[token] = len(vocab)
                word_counts[token] = 1
                
            else:
                word_counts[token] += 1
        
        if rnn:
            line.insert(0, 0)
            line.append(1)

    if rnn:
        return vocab, word_counts, training_data
    
    return vocab, word_counts

def encode_data(data, vocab, word_counts, threshold=None):
    
    encoded_data = []
    empty_vec = np.zeros(len(vocab))
    
    for line in data:
        
        encoded_line = np.copy(empty_vec)
        
        for token in line:
            
            if threshold:
                
                if word_counts[token] > threshold:
                    encoded_line[vocab[token]] += 1
                
                else:
                    continue
                    
            encoded_line[vocab[token]] += 1
            
        encoded_data.append(encoded_line)
        
    return np.array(encoded_data)
        
def preprocess(training_x):
    training_x_processed = processRawDataFromList(training_x)
    return training_x_processed

class rnnDataset(Dataset):
    
    def __init__(self, encoded_x, encoded_y, encoder):
        
        self.encoder = encoder
        self.encoded_x = encoded_x
        self.encoded_y = encoded_y
    
    def hottyY(self, int):
    
        if int == 0:
            return torch.Tensor([1, 0])
        elif int == 1:
            return torch.Tensor([0,1])
    
    def __len__(self):
        
        return len(self.encoded_x)
    
    def __getitem__(self, idx):
        
        x = self.encoded_x[idx]
        y = self.hottyY(self.encoded_y[idx])
        line = self.encoder.decode(x)
        
        return {'x' : x,
                'y' : y,
                'decode' : ' '.join(line)}