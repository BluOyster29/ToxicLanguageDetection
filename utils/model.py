import torch.nn as nn
import torch 

class rnnModel(nn.Module):
    
    def __init__(self, vocab_size, hidden_dim, embedding_dim, output_size,
                 num_layers, pretrained=None, pretrained_vectors=None):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.hidden_size = hidden_dim
        self.output_size = output_size 
        
        if pretrained:
          self.embed = nn.Embedding.from_pretrained(pretrained_vectors,
                                    padding_idx=len(pretrained_vectors)-1)
        
        else:
          self.embed = nn.Embedding(vocab_size, embedding_dim)

        self.lstm = nn.GRU(embedding_dim, hidden_dim, num_layers ,
                           batch_first=True, bidirectional=True)
        self.fc1 = nn.Linear(hidden_dim*2, output_size)


        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        
        embeddings = self.embed(x)
        output, h_n = self.lstm(embeddings)

        concat_output = torch.cat([h_n[0,:, :], h_n[1,:,:]], dim=1)

        output2 = self.fc1(concat_output)


        return self.sigmoid(output2)


