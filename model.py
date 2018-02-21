# _*_ coding:utf-8 _*_
import torch
import torch.nn as nn
from torch.autograd import Variable

class PoetryModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(PoetryModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.embeddings = nn.Embedding(vocab_size, embedding_dim)
        self.lstm = nn.LSTM(embedding_dim, self.hidden_dim,num_layers=2)
        self.linear1 = nn.Linear(self.hidden_dim, vocab_size)

    def forward(self, input,hidden=None):
        seq_len,batch_size = input.size()
        if hidden is None:
            h_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            c_0 = input.data.new(2, batch_size, self.hidden_dim).fill_(0).float()
            h_0,c_0 = Variable(h_0),Variable(c_0)
        else:
            h_0,c_0 = hidden
        embeds = self.embeddings(input)                        # size: (seq_len,batch_size,embeding_dim)
        output, hidden = self.lstm(embeds, (h_0,c_0))            # output size: (seq_len,batch_size,hidden_dim)
        output = self.linear1(output.view(seq_len*batch_size, -1))   # size: (seq_len*batch_size,vocab_size)
        return output,hidden