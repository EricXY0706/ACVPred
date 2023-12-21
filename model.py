import numpy as np
import random
import torch
import torch.nn as nn
import os

def set_seed(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
class Multi_Head_Attention(nn.Module):
    def __init__(self, emb_length, n_heads):
        super(Multi_Head_Attention, self).__init__()
        self.n_heads = n_heads
        self.dim_head = emb_length // self.n_heads
        self.W_Q = nn.Linear(emb_length, n_heads * self.dim_head)
        self.W_K = nn.Linear(emb_length, n_heads * self.dim_head)
        self.W_V = nn.Linear(emb_length, n_heads * self.dim_head)
        self.attention = Dot_Product_Attention()
        self.fc = nn.Linear(n_heads * self.dim_head, emb_length)
        self.LayerNorm = nn.LayerNorm(emb_length)
    def forward(self, x, pad_index):
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)
        Q = Q.view(self.n_heads, -1, self.dim_head)
        K = K.view(self.n_heads, -1, self.dim_head)
        V = V.view(self.n_heads, -1, self.dim_head)
        attn = self.attention(Q, K, V, self.dim_head, pad_index)
        attn = attn.view(-1, self.n_heads * self.dim_head)
        out = self.fc(attn)
        out = out + x
        out = self.LayerNorm(out)
        return out
class Dot_Product_Attention(nn.Module):
    def __init__(self):
        super(Dot_Product_Attention, self).__init__()
    def forward(self, Q, K, V, d, pad_index):
        scores = torch.matmul(Q, K.permute(0, 2, 1)) / d ** 0.5
        scores = scores.masked_fill(pad_index==1, -1e9)
        scores = nn.Softmax(dim=-1)(scores)
        attn = torch.matmul(scores, V)
        return attn
class Feed_Forward_NN(nn.Module):
    def __init__(self, emb_length, hidden, dropout):
        super(Feed_Forward_NN, self).__init__()
        self.fc1 = nn.Linear(emb_length, hidden)
        self.fc2 = nn.Linear(hidden, emb_length)
        self.activate_func = nn.LeakyReLU()
        self.dropout = nn.Dropout(dropout)
        self.LayerNorm = nn.LayerNorm(emb_length)
    def forward(self, x):
        out = self.fc1(x)
        # out = self.dropout(out)
        out = self.activate_func(out)
        out = self.fc2(out)
        out = out + x
        out = self.LayerNorm(out)
        return out
class Encoder_block(nn.Module):
    def __init__(self, emb_length, n_heads, hidden, dropout):
        super(Encoder_block, self).__init__()
        self.attn = Multi_Head_Attention(emb_length, n_heads)
        self.ffnn = Feed_Forward_NN(emb_length, hidden, dropout)
    def forward(self, x, pad_index):
        out = self.attn(x, pad_index)
        out = self.ffnn(out)
        return out
class Encoder(nn.Module):
    def __init__(self, emb_length, n_heads, hidden, dropout, n_encoder):
        super(Encoder, self).__init__()
        self.Layers = nn.ModuleList([Encoder_block(emb_length, n_heads, hidden, dropout) for _ in range(n_encoder)])
    def forward(self, x, pad_index):
        out = x
        for layer in self.Layers:
            out = layer(out, pad_index)
        return out
class Encoder_Classifier(nn.Module):
    def __init__(self, emb_length, n_heads, hidden, dropout, n_encoder):
        super(Encoder_Classifier, self).__init__()
        self.encoder = Encoder(emb_length, n_heads, hidden, dropout, n_encoder)
        self.classifier = nn.Linear(emb_length, 2)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x, pad_index):
        out = self.encoder(x, pad_index)
        zero_index = torch.where(torch.any(pad_index.T != 0, dim=1))[0]
        out[zero_index, :] = 0
        out = torch.sum(out, dim=0) / (out != 0).any(dim=1).sum().item()
        out = self.classifier(out)
        out = nn.Softmax(dim=-1)(out)
        return out