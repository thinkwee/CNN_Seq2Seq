"""自定义一些初始化权重的层，来自Fairseq源码"""

from torch import nn
import math
import torch
import torch.nn.functional as F
from parameters import DEVICE


def Linear(in_features, out_features, dropout=0):
    """Weight-normalized Linear layer (input: N x T x C)"""
    m = nn.Linear(in_features, out_features)
    nn.init.normal_(m.weight,
                    mean=0,
                    std=math.sqrt((1 - dropout) / in_features))
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m)


class MaskConv1d(nn.Conv1d):
    """Masked Conv Kernel for Temporal Convolution"""

    def __init__(self, in_channels, out_channels, kernel_size, stride, padding,
                 dilation):
        super().__init__(in_channels, out_channels, kernel_size, stride,
                         padding, dilation)
        self.mask = torch.tensor([
            0 for _ in range(self.out_channels * self.in_channels *
                             self.kernel_size[0])
        ])
        for idx in range(self.out_channels * self.in_channels *
                         self.kernel_size[0])[::self.kernel_size[0]]:
            for jdx in range(1 + self.kernel_size[0] // 2):
                self.mask[idx + jdx] = 1
        self.mask = self.mask.contiguous().view(
            self.out_channels, self.in_channels,
            self.kernel_size[0]).float().to(DEVICE)

    def forward(self, x):
        return F.conv1d(input=x,
                        weight=self.weight * self.mask,
                        stride=self.stride,
                        padding=self.padding,
                        dilation=self.dilation)

    def show_mask(self):
        return self.mask

    def show_kernel(self):
        return self.weight

    def show_masked_kernel(self):
        return self.mask * self.weight


def WeightNormalizedConv1d(in_channels, out_channels, kernel_size, stride,
                           padding, dropout, dilation):
    """Weight-normalized Conv1d layer"""
    m = nn.Conv1d(in_channels, out_channels, kernel_size, stride, padding,
                  dilation)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


def MaskedWeightNormalizedConv1d(in_channels, out_channels, kernel_size,
                                 stride, padding, dropout, dilation):
    """Masked Weight-normalized Conv1d layer"""
    m = MaskConv1d(in_channels, out_channels, kernel_size, stride, padding,
                   dilation)
    std = math.sqrt((4 * (1.0 - dropout)) / (m.kernel_size[0] * in_channels))
    nn.init.normal_(m.weight, mean=0, std=std)
    nn.init.constant_(m.bias, 0)
    return nn.utils.weight_norm(m, dim=2)


class BeamableMM(nn.Module):
    """This module provides an optimized MM for beam decoding with attention.
    It leverage the fact that the source-side of the input is replicated beam
    times and the target-side of the input is of width one. This layer speeds up
    inference by replacing the inputs {(bsz x 1 x nhu), (bsz x sz2 x nhu)}
    with smaller inputs {(bsz/beam x beam x nhu), (bsz/beam x sz2 x nhu)}.
    """

    def __init__(self, beam_size=None):
        super(BeamableMM, self).__init__()
        self.beam_size = beam_size

    def forward(self, input1, input2):
        if (not self.training and  # test mode
                self.beam_size is not None and  # beam size is set
                input1.dim() == 3 and  # only support batched input
                input1.size(1) == 1  # single time step update
            ):
            bsz, beam = input1.size(0), self.beam_size

            # bsz x 1 x nhu --> bsz/beam x beam x nhu
            input1 = input1[:, 0, :].unfold(0, beam, beam).transpose(2, 1)

            # bsz x sz2 x nhu --> bsz/beam x sz2 x nhu
            input2 = input2.unfold(0, beam, beam)[:, :, :, 0]

            # use non batched operation if bsz = beam
            if input1.size(0) == 1:
                output = torch.mm(input1[0, :, :], input2[0, :, :])
            else:
                output = input1.bmm(input2)
            return output.view(bsz, 1, -1)
        else:
            return input1.bmm(input2)

    def set_beam_size(self, beam_size):
        self.beam_size = beam_size


class AttentionLayer(nn.Module):
    """
        Args:
            x: [B, C, Tg], Decoder convolution block output(after glu)
            target_embedding: [B, Tg, E], Decoder input embeddings
            encoder_out: (attention_key, attention_value)
                attention_key: [B, E, Ts], output from encoder, as key
                attention_value: [B, Ts, E], output from encoder + encoder input embeddings, as value
            encoder_padding_mask: [B, Ts], encoder input mask
        Returns:
            x: [B, Tg, C], attended encoder context + residual of input x
            attn_scores: [B, Tg, Ts], attention weight matrix
    """

    def __init__(self, conv_channels, embed_dim, bmm=None):
        super().__init__()
        # projects from output of convolution to embedding dimension
        self.in_projection = Linear(conv_channels, embed_dim)
        # projects from embedding dimension to convolution size
        self.out_projection = Linear(embed_dim, conv_channels)

        self.bmm = bmm if bmm is not None else torch.bmm

    def forward(self, x, target_embedding, encoder_out, encoder_padding_mask):

        # [B, Tg, C]
        x = x.permute(0, 2, 1)
        residual = x

        # attention
        # [B, Tg, Ts]
        x = (self.in_projection(x) + target_embedding) * math.sqrt(0.5)
        x = self.bmm(x, encoder_out[0])

        # don't attend over padding
        # [B, Tg, Ts]
        if encoder_padding_mask is not None:
            x = x.float().masked_fill(
                encoder_padding_mask.unsqueeze(1), float('-inf')).type_as(
                    x)  # FP16 support: cast to float and back

        # softmax over last dim(Ts)
        # fairseq solution
        # sz = x.size()
        # x = F.softmax(x.view(sz[0] * sz[1], sz[2]), dim=1)
        # x = x.view(sz)

        # why not direct put softmax on dim 2?
        # [B, Tg, Ts]
        attn_scores = F.softmax(x, dim=2)

        # use attention score to weight the attention value
        # [B, Tg, E]
        x = self.bmm(attn_scores, encoder_out[1])

        # scale attention output (respecting potentially different lengths)
        # s = Ts
        s = encoder_out[1].size(1)
        if encoder_padding_mask is None:
            x = x * (s * math.sqrt(1.0 / s))
        else:
            s = s - encoder_padding_mask.type_as(x).sum(
                dim=1, keepdim=True)  # exclude padding
            s = s.unsqueeze(-1)
            x = x * (s * s.rsqrt())

        # project back & add residual
        x = (self.out_projection(x) + residual) * math.sqrt(0.5)
        return x, attn_scores