import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size=10002, embedding_dim=128, n_filters, filter_sizes, output_dim, dropout):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv1d(in_channels = 1,
                                              out_channels = n_filters,
                                              kernel_size = (fs, embedding_dim)) for fs in filter_sizes
                                    ])
        self.fully_Connected = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x = [batch_size, maxLen]
        
        embedding_vec = self.embedding(x)
        # embedded = [batch_size, maxLen, emb_dim]
        
        embedding_vec = embedding_vec.transpose(1, 2)
        #embedded = [batch_size, emb_dim, maxLen]
        
        conv_vecs = [F.relu(conv(embedding_vec)).squeeze(3) for conv in self.convs]
        #conv_vecs = [batch_size, n_filters, maxlen - filter_sizes[n] + 1]
        
        pooled_vecs = [F.max_pool1d(conv_vec, conv_vec.shape[2]).squeeze(2) for conv_vec in conv_vecs]
        #pooled_vec = [batch_size, n_filters]
        
        concat = self.dropout(torch.cat(pooled_vecs, dim = 1))
        #concat = [batch_size, n_filters * len(filter_sizes)]
        
        return self.fully_Connected(concat)