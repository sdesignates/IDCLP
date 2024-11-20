import torch.nn as nn
import torch
from multi_head_attention import MultiHeadAttention, PositionwiseFeedforward


class DrugGeneAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.self_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.encoder_attention = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.fc = nn.Linear(2 * hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        # self attention
        enc_src = enc_src.unsqueeze(1)
        _trg_enc, _ = self.encoder_attention(trg, enc_src, enc_src, src_mask)
        _trg_dec, _ = self.self_attention(trg, trg, trg, trg_mask)
        # _trg = _trg_enc + _trg_dec
        _trg = torch.cat((_trg_enc, _trg_dec), dim=2)
        _trg = torch.relu(self.fc(_trg))
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.layer_norm(trg + self.dropout(_trg))
        return trg, None

class DrugCellAttentionLayer(nn.Module):
    def __init__(self, hid_dim, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.layer_norm = nn.LayerNorm(hid_dim)
        self.encoder_attention1 = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.encoder_attention2 = MultiHeadAttention(hid_dim, n_heads, dropout, device)
        self.positionwise_feedforward = PositionwiseFeedforward(hid_dim, pf_dim, dropout)
        self.fc = nn.Linear(2 * hid_dim, hid_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len, hid dim]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        # self attention
        _trg_enc, _ = self.encoder_attention1(trg, enc_src, enc_src, src_mask)
        _trg_dec, _ = self.encoder_attention2(enc_src, trg, trg, trg_mask)
        # _trg = _trg_enc + _trg_dec
        _trg = torch.cat((_trg_enc, _trg_dec), dim=2)
        _trg = torch.relu(self.fc(_trg))
        trg = trg.unsqueeze(1)
        trg = self.layer_norm(trg + self.dropout(_trg))
        _trg = self.positionwise_feedforward(trg)
        trg = self.layer_norm(trg + self.dropout(_trg))
        trg = trg.squeeze(1)
        return trg, None

class DrugCellAttention(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([DrugCellAttentionLayer(hid_dim,n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        # trg = [batch size, trg len, hid dim]
        if src_mask != None:
            src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]
        return output, attention

class DrugGeneAttention(nn.Module):
    def __init__(self, output_dim, hid_dim, n_layers, n_heads, pf_dim, dropout, device):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList([DrugGeneAttentionLayer(hid_dim,n_heads, pf_dim, dropout, device)
                                     for _ in range(n_layers)])
        self.fc_out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, trg, enc_src, trg_mask, src_mask):
        # trg = [batch size, trg len]
        # enc_src = [batch size, src len, hid dim]
        # trg_mask = [batch size, trg len]
        # src_mask = [batch size, src len]
        # trg = [batch size, trg len, hid dim]
        if src_mask != None:
            src_mask = src_mask.unsqueeze(1).unsqueeze(2)
        for layer in self.layers:
            trg, attention = layer(trg, enc_src, trg_mask, src_mask)
        # trg = [batch size, trg len, hid dim]
        # attention = [batch size, n heads, trg len, src len]
        output = self.fc_out(trg)
        # output = [batch size, trg len, output dim]
        return output, attention