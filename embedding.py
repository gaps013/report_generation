import torch
from torch import nn

class WordAndPositionalEmbedding(nn.Module):
    def __init__(self, device, vocab_size=1000,
                 embedding_dim=512, drop_out=0.1, max_sequence_length=256, padding_idx=0):
        super(WordAndPositionalEmbedding, self).__init__()
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.drop_out = drop_out
        self.padding_idx = padding_idx
        self.max_sequence_length = max_sequence_length
        self. wte = nn.Embedding(num_embeddings=self.vocab_size, embedding_dim=self.embedding_dim,
                                 padding_idx=self.padding_idx)
        self.wtp = nn.Embedding(num_embeddings=self.max_sequence_length, embedding_dim=self.embedding_dim)
        self.layer_nor = nn.LayerNorm(embedding_dim)
        self.drop_out = nn.Dropout(p=self.drop_out)
        self.device = device
    def forward(self, inputs):
        pos_ids = torch.arange(0, inputs.size(-1)).unsqueeze(0).to(self.device)
        embeddings = self.layer_nor(self.wte(inputs)+self.wtp(pos_ids))
        embeddings = self.drop_out(embeddings)
        # Zero-out embeddings for positions which have padding tokens.
        # shape: (batch_size, max_caption_length, 1)
        token_mask = (inputs != self.padding_idx).unsqueeze(-1)

        # shape: (batch_size, max_caption_length, hidden_size)
        embeddings = embeddings * token_mask.type(embeddings.dtype)
        return embeddings