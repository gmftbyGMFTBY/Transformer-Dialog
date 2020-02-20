'''
@Description: define some transformer models
@Author: bairongz (baiyuu.cs@gmail.com)
@Date: 2019-09-10 15:56:47
@LastEditTime : 2019-12-21 15:57:06
'''

import torch
from torch import nn
import numpy as np
import constants as C
from common_layers import *


class Transformer(nn.Module):
    def __init__(self, n_vocab, d_model=512, n_head=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1, max_len=512, max_turn=256, share_word_embedding=True, n_dec_vocab=-1):
        """Transformer
        
        Arguments:
            n_vocab {int} -- # of vocabulary
        
        Keyword Arguments:
            d_model {int} -- dimension of hidden state (default: {512})
            n_head {int} -- # of heads used in multi-head attention (default: {8})
            num_encoder_layers {int} -- # of transformer encoder layers (default: {6})
            num_decoder_layers {int} -- # of transformer decoder blocks (default: {6})
            dim_feedforward {int} -- dimension of hidden layer of position wise feed forward layer(default: {2048})
            dropout {float} -- dropout rate (default: {0.1})
            max_len {int} -- max input length (default: {512})
            max_turn {int} -- max input turn (default: {256})
            share_word_embedding {bool} -- share word embedding between encoder and decoder
        """
        super().__init__()
        self.n_vocab= n_vocab
        self.n_dec_vocab = n_vocab if n_dec_vocab==-1 else n_dec_vocab
        self.enc_word_embed = nn.Embedding(n_vocab, d_model, padding_idx=C.PAD)
        self.pos_embed  = nn.Embedding(max_len+1, d_model, padding_idx=C.PAD)
        self.turn_embed = nn.Embedding(max_turn+1, d_model, padding_idx=C.PAD)
        if share_word_embedding:
            self.dec_word_embed = self.enc_word_embed
        else:
            self.dec_word_embed = nn.Embedding(n_dec_vocab, d_model, padding_idx=C.PAD)

        encoder_layer = nn.TransformerEncoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        encoder_norm =nn.LayerNorm(d_model)
        self.encoder = nn.TransformerEncoder(encoder_layer, num_encoder_layers, encoder_norm)

        decoder_layer = nn.TransformerDecoderLayer(d_model, nhead=n_head, dim_feedforward=dim_feedforward, dropout=dropout)
        decoder_norm = nn.LayerNorm(d_model)
        self.decoder = nn.TransformerDecoder(decoder_layer, num_decoder_layers, decoder_norm)

        self.proj = nn.Linear(d_model, self.n_dec_vocab, bias=False)


    @staticmethod
    def get_masks(src, trg, PAD=0):
        """generate masks based on inputs and targets
        
        Arguments:
            src {torch.LongTensor} -- input mini-batch in shape (L, B)
            trg {torch.LongTensor} -- target mini-batch in shape (L, B)
        
        Keyword Arguments:
            PAD {int} -- padding value (default: {0})
        
        Returns:
            masks can be directly passed into encode/decode's forward method
        """
        S, S_B = src.shape
        T, T_B = trg.shape
        assert S_B == T_B, "Batch Size of source input and target input inconsistent! %d != %d"%(S_B, T_B)

        trg_mask = nn.Transformer.generate_square_subsequent_mask(T, T)
        src_key_padding_mask = (src==PAD).permute(1,0)
        trg_key_padding_mask = (trg==PAD).permute(1,0)
        memory_key_padding_mask = (src==PAD).permute(1,0)

        return trg_mask, src_key_padding_mask, trg_key_padding_mask, memory_key_padding_mask

    def encode(self, src, src_pos, src_turn=None, src_mask=None, src_key_padding_mask=None):
        """encode step of transformer
        
        Arguments:
            src {torch.LongTensor} -- input mini-batch in shape (L_S, B)
            src_pos {torch.LongTensor} -- input position ids in range (1, L), padded position filled with 0
        
        Keyword Arguments:
            src_turn {torch.LongTensor} -- turn ids in range (1, T) (default: {None})
        
        Returns:
            torch.Tensor -- output of transformer encoder in shape (L_S, B, H)
        """

        if src_turn is not None:
            src_embed = self.enc_word_embed(src) + self.pos_embed(src_pos) + self.turn_embed(src_turn)
        else:
            src_embed = self.enc_word_embed(src) + self.pos_embed(src_pos)

        memory = self.encoder(src_embed,
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)

        return memory

    def decode(self, trg, trg_pos, src, memory, trg_mask=None, memory_mask=None, trg_key_padding_mask=None, memory_key_padding_mask=None):
        """decode step of transformer
        
        Arguments:
            trg {torch.LongTensor} -- target mini-batch in shape (L_T, B)
            trg_pos {torch.LongTensor} -- target position ids in range (1, L), padded position filled with 0
            src {torch.LongTensor} -- input mini-batch in shape (L_S, B)
            memory {torch.Tensor} -- output of transformer encoder in shape (L_S, B, H)
        
        Returns:
            torch.Tensor -- output of decoder in shape (L_T, B, H)
        """

        trg_embed = self.dec_word_embed(trg) + self.pos_embed(trg_pos)
        dec_output = self.decoder(trg_embed, memory,
                              tgt_mask=trg_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=trg_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        return dec_output

        
    def forward(self, src, trg, src_pos, trg_pos, src_turn=None,src_mask=None, trg_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                trg_key_padding_mask=None, memory_key_padding_mask=None):
        """forward computation for Transformer
        
        Arguments:
            src {torch.LongTensor} -- input mini-batch in shape (L_S, B)
            trg {torch.LongTensor} -- target mini-batch in shape (L_T, B)
            src_pos {torch.LongTensor} -- input position ids in range (1, L), padded position filled with 0
            trg_pos {torch.LongTensor} -- target position ids in range (1, L), padded position filled with 0
        
        Keyword Arguments:
            src_turn {torch.LongTensor} -- turn ids in range (1, T) (default: {None})
        
        Returns:
            torch.Tensor -- logits in shape (L_T, V)
        """

        if src_turn is not None:
            src_embed = self.enc_word_embed(src) + self.pos_embed(src_pos) + self.turn_embed(src_turn)
        else:
            src_embed = self.enc_word_embed(src) + self.pos_embed(src_pos)

        trg_embed = self.dec_word_embed(trg) + self.pos_embed(trg_pos)

        memory = self.encoder(src_embed,
                              mask=src_mask,
                              src_key_padding_mask=src_key_padding_mask)

        output = self.decoder(trg_embed, memory,
                              tgt_mask=trg_mask,
                              memory_mask=memory_mask,
                              tgt_key_padding_mask=trg_key_padding_mask,
                              memory_key_padding_mask=memory_key_padding_mask)

        logits = self.proj(output)
        return logits.view(-1, logits.size(2))

class UniversalTransformer(nn.Module):
    pass
