import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Tuple

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]

class DateTransformer(nn.Module):
    def __init__(self, in_vocab_size: int, out_vocab_size: int, d_model: int = 256, nhead: int = 8, num_layers: int = 3, dropout: float = 0.1):
        super().__init__()
        self.embedding_in = nn.Embedding(in_vocab_size, d_model)
        self.embedding_out = nn.Embedding(out_vocab_size, d_model)
        self.pos_encoder = PositionalEncoding(d_model)
        
        self.transformer = nn.Transformer(d_model=d_model, nhead=nhead, 
                                          num_encoder_layers=num_layers, 
                                          num_decoder_layers=num_layers,
                                          dim_feedforward=d_model*4,
                                          dropout=dropout,
                                          batch_first=True)
        self.fc_out = nn.Linear(d_model, out_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        src_emb = self.pos_encoder(self.embedding_in(src))
        tgt_emb = self.pos_encoder(self.embedding_out(tgt))
        tgt_seq_len = tgt.size(1)
        tgt_mask = self.transformer.generate_square_subsequent_mask(tgt_seq_len).to(src.device)
        out = self.transformer(src_emb, tgt_emb, tgt_mask=tgt_mask)
        return self.fc_out(out)

class Seq2SeqLSTM(nn.Module):
    def __init__(self, in_vocab_size: int, out_vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.emb_in = nn.Embedding(in_vocab_size, hidden_dim)
        self.emb_out = nn.Embedding(out_vocab_size, hidden_dim)
        self.encoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.decoder = nn.LSTM(hidden_dim, hidden_dim, num_layers=2, batch_first=True, dropout=0.1)
        self.fc = nn.Linear(hidden_dim, out_vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        _, (hidden, cell) = self.encoder(self.emb_in(src))
        out, _ = self.decoder(self.emb_out(tgt), (hidden, cell))
        return self.fc(out)

class ConditionalVAE(nn.Module):
    def __init__(self, in_vocab_size: int, out_vocab_size: int, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.cond_emb = nn.Embedding(in_vocab_size, hidden_dim)
        self.tgt_emb = nn.Embedding(out_vocab_size, hidden_dim)
        self.encoder_rnn = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder_rnn = nn.GRU(hidden_dim * 2 + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, out_vocab_size)

    def encode(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        c = self.cond_emb(src).mean(dim=1, keepdim=True).repeat(1, tgt.size(1), 1)
        x = self.tgt_emb(tgt)
        _, h = self.encoder_rnn(torch.cat([x, c], dim=-1))
        h = h.squeeze(0)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        c = self.cond_emb(src).mean(dim=1, keepdim=True).repeat(1, tgt.size(1), 1)
        z = z.unsqueeze(1).repeat(1, tgt.size(1), 1)
        dec_in = torch.cat([self.tgt_emb(tgt), z, c], dim=-1)
        out, _ = self.decoder_rnn(dec_in)
        return self.fc_out(out)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mu, logvar = self.encode(src, tgt)
        z = self.reparameterize(mu, logvar)
        recon_x = self.decode(z, src, tgt)
        return recon_x, mu, logvar

class ConditionalAE(nn.Module):
    def __init__(self, in_vocab_size: int, out_vocab_size: int, hidden_dim: int = 256, latent_dim: int = 64):
        super().__init__()
        self.cond_emb = nn.Embedding(in_vocab_size, hidden_dim)
        self.tgt_emb = nn.Embedding(out_vocab_size, hidden_dim)
        self.encoder_rnn = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.fc_latent = nn.Linear(hidden_dim, latent_dim)
        
        self.decoder_rnn = nn.GRU(hidden_dim * 2 + latent_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, out_vocab_size)

    def encode(self, src: torch.Tensor) -> torch.Tensor:
        _, h = self.encoder_rnn(self.cond_emb(src))
        return self.fc_latent(h.squeeze(0))

    def decode(self, z: torch.Tensor, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        c = self.cond_emb(src).mean(dim=1, keepdim=True).repeat(1, tgt.size(1), 1)
        z_rep = z.unsqueeze(1).repeat(1, tgt.size(1), 1)
        dec_in = torch.cat([self.tgt_emb(tgt), z_rep, c], dim=-1)
        out, _ = self.decoder_rnn(dec_in)
        return self.fc_out(out)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        z = self.encode(src)
        return self.decode(z, src, tgt)

class Generator(nn.Module):
    def __init__(self, in_vocab_size: int, out_vocab_size: int, hidden_dim: int = 256, noise_dim: int = 64):
        super().__init__()
        self.cond_emb = nn.Embedding(in_vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim + noise_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, out_vocab_size)

    def forward(self, src: torch.Tensor, z: torch.Tensor, seq_len: int) -> torch.Tensor:
        c = self.cond_emb(src).mean(dim=1, keepdim=True).repeat(1, seq_len, 1)
        z = z.unsqueeze(1).repeat(1, seq_len, 1)
        out, _ = self.rnn(torch.cat([c, z], dim=-1))
        logits = self.fc_out(out)
        return F.gumbel_softmax(logits, tau=1.0, hard=True, dim=-1)

class Discriminator(nn.Module):
    def __init__(self, in_vocab_size: int, out_vocab_size: int, hidden_dim: int = 256):
        super().__init__()
        self.cond_emb = nn.Embedding(in_vocab_size, hidden_dim)
        self.seq_proj = nn.Linear(out_vocab_size, hidden_dim)
        self.rnn = nn.GRU(hidden_dim * 2, hidden_dim, batch_first=True)
        self.fc_out = nn.Sequential(nn.Linear(hidden_dim, 1), nn.Sigmoid())

    def forward(self, src: torch.Tensor, seq_one_hot: torch.Tensor) -> torch.Tensor:
        c = self.cond_emb(src).mean(dim=1, keepdim=True).repeat(1, seq_one_hot.size(1), 1)
        x = self.seq_proj(seq_one_hot)
        _, h = self.rnn(torch.cat([x, c], dim=-1))
        return self.fc_out(h.squeeze(0))