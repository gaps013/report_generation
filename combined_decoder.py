import torch
from torch import nn
from embedding import WordAndPositionalEmbedding
class CombinedDecoder(nn.Module):
    def __init__(self, device, vocab_size=10000, embedding_dim=512, num_layers=6,
                 attention_heads=3, drop_out=0.1, max_sequence_length=256, padding_idx=0):
        super(CombinedDecoder, self).__init__()
        self.device = device
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        self.attention_heads = attention_heads
        self.drop_out = drop_out
        self.max_sequence_length = max_sequence_length
        self.padding_idx = padding_idx
        self.device = device
        self.embedding = WordAndPositionalEmbedding(device=device, vocab_size=self.vocab_size,
                                                    embedding_dim=self.embedding_dim, drop_out=self.drop_out,
                                                    max_sequence_length=self.max_sequence_length, padding_idx=self.padding_idx)

        self.decoder_layer = nn.TransformerDecoderLayer(d_model=self.embedding_dim, nhead=self.attention_heads,
                                                        dropout=self.drop_out, dim_feedforward=4*self.embedding_dim)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, self.num_layers)
        self.output = nn.Linear(self.embedding_dim, self.vocab_size)
        self.output.weight = self.embedding.wte.weight

    def forward(self, image_features, report, report_length):
        batch_size, max_sequence_legth = report.size()
        ones = torch.ones_like(report)
        report_mask = report_length.unsqueeze(1) < ones.cumsum(dim=1)
        # shape: (batch_size, max_caption_length, textual_feature_size)
        report_embeddings = self.embedding(report)
        report_embeddings = report_embeddings.transpose(0, 1)
        image_features = image_features.transpose(0, 1)
        # shape: (max_caption_length, batch_size, hidden_size)
        textual_features = self.transformer_decoder(
            report_embeddings,
            image_features,
            tgt_key_padding_mask=report_mask,
        )
        # Undo the transpose and bring batch to dim 0.
        # shape: (batch_size, max_caption_length, hidden_size)
        textual_features = textual_features.transpose(0, 1)

        # shape: (batch_size, max_caption_length, vocab_size)
        output_logits = self.output(textual_features)
        return output_logits
