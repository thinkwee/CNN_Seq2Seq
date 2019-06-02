"""Convolutional Sequence to Sequence Learning Implementation by Liu Wei"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import _pickle as pickle
from parameters import DEVICE
from .layers import Linear, MaskedWeightNormalizedConv1d, WeightNormalizedConv1d, AttentionLayer
from .grad_multiply import GradMultiply
from .positional_embed import PositionalEmbedding
from parameters import *
import numpy as np


class Encoder(nn.Module):
    """
        Args:
            input: [B, Ts], one hot presentation of Original Document(Encoder Input)
        Returns:
            outputs: [B, E, Ts], document representation encoded by encoder , as attention key
            attention_value: [B, Ts, E], encoder input embedding + encoder output, as attention value
            encoder_padding_mask: [B, Ts], encoder mask
    """

    def __init__(self):
        super(Encoder, self).__init__()

        # scalar
        self.vocab_size = VOCAB_SIZE_SRC
        self.embedding_size = EMBED_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.channels = HIDDEN_SIZE
        self.kernel_size = KERNEL_SIZE_ENC
        self.layers = ENC_LAYERS
        self.stride = 1
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(DEVICE)
        self.timesteps = SRC_LENGTH
        self.padding_idx = PAD_token
        self.dilation = 1
        self.training = True
        self.num_attention_layers = DEC_LAYERS

        # embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embed_positions = PositionalEmbedding(
            self.timesteps + 1,
            self.embedding_size,
            self.padding_idx,
        )

        # other layers
        self.affine_start = Linear(self.embedding_size,
                                   self.hidden_size,
                                   dropout=DROPOUT_RATIO)
        self.affine_end = Linear(self.hidden_size,
                                 self.embedding_size,
                                 dropout=DROPOUT_RATIO)
        self.softmax = nn.Softmax()
        self.dropout = nn.Dropout(p=DROPOUT_RATIO)
        self.mapping = Linear(self.hidden_size, 2 * self.hidden_size)

        # convolutional blocks
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        for idx in range(self.layers):
            """use interlace dilation like [1,2,1,2,1,2,1,2,1,2,1,2,1...]"""
            if idx % 2 == 0:
                self.dilation = 1
            else:
                self.dilation = 2
            self.padding = self.dilation * (self.kernel_size - 1) // 2
            self.conv_list.append(
                WeightNormalizedConv1d(self.channels, self.channels * 2,
                                       self.kernel_size, self.stride,
                                       self.padding, DROPOUT_RATIO,
                                       self.dilation).to(DEVICE))
            self.bn_list.append(nn.BatchNorm1d(self.hidden_size * 2))

    def forward(self, encoder_input):
        # to device
        encoder_input = encoder_input.to(DEVICE)

        # used to mask padding in encoder_input
        # [B, Ts]
        encoder_padding_mask = encoder_input.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None

        # [B, Ts, E]
        embeddings_enc = self.embedding(encoder_input) + self.embed_positions(
            encoder_input)
        embeddings_enc = self.dropout(embeddings_enc)

        # [B, Ts, C]
        conv_input = self.affine_start(embeddings_enc)

        for i in range(self.layers):
            # mask convolution block input in each block
            # [B, Ts, C]
            if encoder_padding_mask is not None:
                conv_input = conv_input.masked_fill(
                    encoder_padding_mask.unsqueeze(-1), 0).permute(0, 2, 1)
            else:
                conv_input = conv_input.permute(0, 2, 1)

            # WeightNormalizedConv1d
            # [B, 2 * C, Ts]
            conv_input = self.dropout(conv_input)
            conv_output = self.conv_list[i](conv_input)

            # Batch normalization
            # [B, 2 * C, Ts]
            conv_output = self.bn_list[i](conv_output)

            # Gated Linear Unit
            # [B, C, Ts]
            conv_output = F.glu(conv_output, dim=1)

            # residual connection
            # [B, C, Ts]
            conv_output = (conv_output + conv_input) * self.scale

            # for next residual connection
            # [B, C, Ts]
            conv_input = conv_output.permute(0, 2, 1)

        # [B, Ts, E]
        outputs = self.affine_end(conv_output.permute(0, 2, 1))

        # mask last block output after projection
        # [B, Ts, E]
        if encoder_padding_mask is not None:
            outputs = outputs.masked_fill(encoder_padding_mask.unsqueeze(-1),
                                          0)

        # scale gradients (this only affects backward, not forward)
        outputs = GradMultiply.apply(outputs,
                                     1.0 / (2.0 * self.num_attention_layers))

        # value for attention: encoder output + encoder input embeddings
        attention_value = (outputs + embeddings_enc) * self.scale

        return outputs.permute(0, 2, 1), attention_value, encoder_padding_mask

    def load_pretrained_vectors(self, path):
        """ Load pretrained embeddings as init value"""
        if path is not None:
            pretrained = pickle.load(open(path, "rb"))
            self.embedding.weight.data.copy_(pretrained)
            self.embedding.requires_grad = True


class Decoder(nn.Module):
    """
        Teacher Forcing Parallel Decoder
        generate state of all timesteps at once.
        Args:
            attention_value: [B, Ts, E], output from Encoder + Encoder embedding input as attention value
            encoder_output: [B, Ts, C], output from Encoder as attention key
            decoder_input: [B, Tg], gold summary input for teacher forcing learning ,  shift right and add SOS 
        return:
            outputs: [B, Tg, C], decoded presentation for all timesteps, then for each timestep we share a linear layer to
            project it on vocabulary size
    """

    def __init__(self):
        super(Decoder, self).__init__()

        # scalar
        self.vocab_size = VOCAB_SIZE_TGT
        self.embedding_size = EMBED_SIZE
        self.hidden_size = HIDDEN_SIZE
        self.channels = HIDDEN_SIZE
        self.kernel_size = KERNEL_SIZE_DEC
        self.layers = DEC_LAYERS
        self.stride = 1
        self.dilation = 1
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(DEVICE)
        self.timesteps = TGT_LENGTH
        self.padding_idx = PAD_token
        self.padding = self.dilation * (self.kernel_size - 1) // 2
        self.training = True
        self.te_ratio = 0.5

        # embeddings
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_size)
        self.embed_positions = PositionalEmbedding(
            self.timesteps + 1,
            self.embedding_size,
            self.padding_idx,
        )

        # other layers
        self.affine_start = Linear(self.embedding_size,
                                   self.hidden_size,
                                   dropout=DROPOUT_RATIO)
        self.mapping = Linear(self.hidden_size, 2 * self.hidden_size)
        self.projections = Linear(self.hidden_size, self.vocab_size)
        self.dropout = nn.Dropout(p=DROPOUT_RATIO)
        self.softmax = nn.Softmax(dim=2)

        # convolution block
        self.conv_list = nn.ModuleList()
        self.bn_list = nn.ModuleList()
        self.attn_list = nn.ModuleList()
        for _ in range(self.layers):
            self.padding = self.dilation * (self.kernel_size - 1) // 2
            self.conv_list.append(
                MaskedWeightNormalizedConv1d(self.channels, self.channels * 2,
                                             self.kernel_size, self.stride,
                                             self.padding, DROPOUT_RATIO,
                                             self.dilation).to(DEVICE))
            self.attn_list.append(
                AttentionLayer(self.channels, self.embedding_size))
            self.bn_list.append(nn.BatchNorm1d(self.hidden_size * 2))

    def forward(self, attention_value, attention_key, encoder_padding_mask,
                decoder_input):
        # to device
        decoder_input = decoder_input.to(DEVICE)

        # used to mask padding in input
        # [B, Tg]
        decoder_padding_mask = decoder_input.eq(self.padding_idx)
        if not decoder_padding_mask.any():
            decoder_padding_mask = None

        # [B, Tg, E]
        embeddings_dec = self.embedding(decoder_input) + self.embed_positions(
            decoder_input)
        embeddings_dec = self.dropout(embeddings_dec)

        # [B, Tg, C]
        conv_input = self.affine_start(embeddings_dec)

        # record average attention scores among all decoder layers
        avg_attn_scores = None

        for i in range(self.layers):
            # mask convolution block input in each block
            # [B, C, Tg]
            if decoder_padding_mask is not None:
                conv_input = conv_input.masked_fill(
                    decoder_padding_mask.unsqueeze(-1), 0).permute(0, 2, 1)
            else:
                conv_input = conv_input.permute(0, 2, 1)

            # MaskedWeightNormalizedConv1d
            # [B, 2 * C, Tg]
            conv_input = self.dropout(conv_input)
            conv_output = self.conv_list[i](conv_input)

            # Batch normalization
            conv_output = self.bn_list[i](conv_output)

            # [B, C, Tg]
            conv_output = F.glu(conv_output, dim=1)

            # Do MultiHop Attention
            # conv_output [B, Tg, C]
            # attn_scores [B, Tg, Ts]
            conv_output, attn_scores = self.attn_list[i](
                conv_output, embeddings_dec, (attention_key, attention_value),
                encoder_padding_mask)
            if avg_attn_scores is None:
                avg_attn_scores = attn_scores
            # else:
            #     avg_attn_scores.add_(attn_scores)

            # [B, C, Tg]
            conv_output = (conv_output.permute(0, 2, 1) +
                           conv_input) * self.scale

            # [B, Tg, C]
            conv_input = conv_output.permute(0, 2, 1)

        # [B, Tg, vocab_size]
        vocab_score = self.projections(conv_output.permute(0, 2, 1))
        return vocab_score, avg_attn_scores

    def load_pretrained_vectors(self, path):
        """Load pretrained embeddings as init value"""
        if path is not None:
            pretrained = pickle.load(open(path, "rb"))
            self.embedding.weight.data.copy_(pretrained)
            self.embedding.requires_grad = True


class IncrementalDecoder(Decoder):
    """
    No Tearcher Forcing, Step by Step (Incremental) Decoder.
    generate decode states step by step , use last step output or gold decoder input as next step input, 
    which is determined by Teacher Forcing Ratio, known as Scheduled Sampling
    this class should inherit from Decoder because they share same weight.
        Args:
            encoder_input_embed: [B, Ts, E], output from Encoder
            encoder_output: [B, Ts, C], output from Encoder
            decoder_input: [B, src_tgt], gold summary input for teacher forcing learning
        return:
            outputs: [B, Tg, C], decoded presentation for all timesteps, then for each timestep we share a linear layer to
            project it on vocabulary size
    """

    def __init__(self):
        super(IncrementalDecoder, self).__init__()

        # in incremental mode we do not convolute the entire sentence at once
        # so we change the padding to zero
        # instead we will pad it manually
        for i in range(self.layers):
            self.conv_list[i].padding = 0

    def forward(self, attention_value, encoder_output, decoder_input_gold,
                decoder_gold, te_ratio):

        # init decoder input
        # we add STR(which is 3 in one-hot) to the start of input then pad
        # we just pad it in the first layer , so the padding_size should consider kernel_size and layers
        # here the input are one-hot vectors so we just pad zeros vectors
        # [B, seq_len_tgt_pad]
        decoder_input_gold = decoder_input_gold.to(DEVICE)
        decoder_gold = decoder_gold.to(DEVICE)

        batch_size = decoder_input_gold.size()[0]
        decoder_input = torch.zeros(batch_size, TGT_LENGTH).long().to(DEVICE)
        decoder_input[:, 0:1] = torch.Tensor([3 for _ in range(batch_size)
                                              ]).view(batch_size, -1)

        pad_size = self.layers * (self.kernel_size - 1) // 2
        pad = torch.zeros(batch_size, pad_size).long().to(DEVICE)

        decoder_input = torch.cat([pad, decoder_input, pad], dim=1)
        decoder_input_gold = torch.cat([pad, decoder_input_gold, pad], dim=1)

        # for calculating loss
        cross_entrophy_loss = torch.nn.CrossEntropyLoss(ignore_index=PAD_token,
                                                        reduction="sum")
        train_predicted = np.zeros(shape=(0, decoder_input.size()[0]))

        # kernel size window in first layer
        window = 1 + self.layers * (self.kernel_size - 1)

        # # temp for test
        # id2word = pickle.load(open("./data/id2word.dat", "rb"))
        # flag=""

        # step by step
        # optimize GPU memory here
        for cur in range(TGT_LENGTH):

            # make a deep copy to prevent variables needed for gradient computation been modified by an inplace operation
            temp_decoder_input = decoder_input.clone()

            # print(' '.join([id2word[id] for id in temp_decoder_input[0].cpu().numpy()]) + flag)

            # each step we only fetch part of sentence(kernel size window)
            # [B, seq_len_tgt_window]
            partial_decoder_input = temp_decoder_input[:, cur:cur + window]

            # [B, seq_len_tgt_window, E]
            embeddings_dec = self.embedding(partial_decoder_input)

            # [B, seq_len_tgt_window, C]
            # outputs = F.dropout(outputs, p=self.dropout, training=self.training)
            conv_input = self.affine_start(embeddings_dec).permute(0, 2, 1)
            # print("after affine |   ", outputs.size())

            # use part of input to do multilayer convolution decode
            # after each layer the seq_len_tgt_windows will get smaller
            # in the last layer the seq_len_tgt_windows = 1, which equals to current output status (before a linear project to vocab)
            for i in range(self.layers):
                # print("------  dec layer %d ------" % i)

                # [B, 2 * C, seq_len_tgt_window]
                conv_output = self.conv_list[i](conv_input)

                # B normalization
                conv_output = self.bn_list[i](conv_output)

                # [B, seq_len_tgt_window, C]
                conv_output = F.glu(conv_output, dim=1)

                conv_input = conv_output

            # do attention only in last layer
            # [B, 1, Ts]
            _attn_matrix = torch.bmm(conv_output.permute(0, 2, 1),
                                     encoder_output.permute(0, 2, 1))
            attn_matrix = self.softmax(_attn_matrix)

            # [B, 1, C]
            attn_weighted_context = torch.bmm(attn_matrix, attention_value)

            # [B, 1, C]
            conv_output = attn_weighted_context.contiguous().squeeze()

            # [B, 1, vocab_size]
            vocab_score = self.projections(conv_output)

            # current output word idx
            # [B, 1]
            cur_vocab_predicted = torch.max(vocab_score, 1)[1]

            # get model predicted output when training
            train_predicted = np.row_stack(
                (train_predicted, cur_vocab_predicted.cpu().numpy()))

            # choose gold decoder input or decoded sentence as input based on teacher forcing ratio
            if np.random.uniform() < te_ratio:
                # teacher forcing
                decoder_input[:, pad_size + cur + 1:pad_size + cur +
                              2] = decoder_input_gold[:, pad_size + cur +
                                                      1:pad_size + cur + 2]
                # flag = "yes"
            else:
                # use last decoded word
                decoder_input[:, pad_size + cur + 1:pad_size + cur +
                              2] = cur_vocab_predicted.contiguous().view(
                                  -1, 1)
                # flag = ""

            # # choose gold decoder input or decoded sentence as input based on teacher forcing ratio and decoded length
            # if cur / TGT_LENGTH < te_ratio:
            #     # teacher forcing
            #     decoder_input[:, pad_size + cur + 1:pad_size + cur + 2] = decoder_input_gold[:,pad_size + cur + 1:pad_size + cur + 2]
            # else:
            #     # use last decoded word
            #     decoder_input[:, pad_size + cur + 1:pad_size + cur + 2] = cur_vocab_predicted.contiguous().view(-1, 1)

            # make select mask to select batched target word onehot on current word position
            # [B, 1]
            cur_select_batch = torch.tensor([
                cur for _ in range(vocab_score.size()[0])
            ]).contiguous().view(vocab_score.size()[0], -1).to(DEVICE)

            # select batched target word onehot based on select mask and squeeze it
            # [B]
            cur_tgt_batch = torch.gather(decoder_gold, 1,
                                         cur_select_batch).squeeze()

            # calculate loss per word position
            # [B, 1]
            loss = cross_entrophy_loss(vocab_score, cur_tgt_batch)

            # # get the length mask based on current idx and sentence length
            # # [B, 1]
            # length_mask = decoder_len_mask.ge(cur + 1).contiguous().view(
            #     -1, 1).float()

            # # filter the loss with filter mask
            # # [B, 1]
            # if cur == 0:
            #     loss_total = torch.mul(loss, length_mask).sum()
            # else:
            #     loss_total += torch.mul(loss, length_mask).sum()

            if cur == 0:
                loss_total = loss.sum()
            else:
                loss_total += loss.sum()

        return loss_total, train_predicted.T

    def load_pretrained_vectors(self, path):
        """Load pretrained embeddings as init value"""
        if path is not None:
            pretrained = pickle.load(open(path, "rb"))
            self.embedding.weight.data.copy_(pretrained)
            self.embedding.requires_grad = True