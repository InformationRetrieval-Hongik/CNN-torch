import torch
import torch.nn as nn
import torch.nn.functional as F

class CNN(nn.Module):
    def __init__(self, vocab_size=10002, embedding_dim=128, n_filters = 100, filter_sizes = [3, 4, 5], output_dim = 1, dropout = 0.5):
        super(CNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.convs = nn.ModuleList([
                                    nn.Conv2d(in_channels = 1,
                                              out_channels = n_filters,
                                              kernel_size = (fs, embedding_dim)) for fs in filter_sizes
                                    ])
        self.fully_Connected = nn.Linear(len(filter_sizes) * n_filters, output_dim)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x):
        # x = [batch_size, maxLen]
        
        embedding_vec = self.embedding(x)
        # embedding_vec = [batch_size, maxLen, emb_dim]
        embedding_vec = embedding_vec.unsqueeze(1)
        # embedding_vec = embedding_vec.permute(0, 2, 1)
        # # embedding_vec = [batch_size, 1, maxLen, emb_dim]
        
        conv_vecs = [F.relu(conv(embedding_vec)).squeeze(3) for conv in self.convs]
        # conv_vecs = [batch_size, n_filters, maxlen - filter_sizes[n] + 1]
        
        pooled_vecs = [F.max_pool1d(conv_vec, conv_vec.shape[2]).squeeze(2) for conv_vec in conv_vecs]
        # pooled_vec = [batch_size, n_filters]
        
        concat = self.dropout(torch.cat(pooled_vecs, dim = 1))
        # concat = [batch_size, n_filters * len(filter_sizes)]
        
        out = self.sigmoid(self.fully_Connected(concat))
        # out = [batch_size, 1]
        
        return out