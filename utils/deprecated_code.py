'''
    废弃没用的一些snippet
'''

# import torch
# from parameters import DEVICE
# # 创建遮罩张量来遮罩decoder输入
# def create_mask_tensor(batch_size,length):
#     mask_tensor = torch.ones(0)
#     batch_size = batch_size
#     len_sentence = length
#     mask = torch.ones(len_sentence)
#     for _ in range(batch_size):
#         for i in range(1, len_sentence + 1):
#             masked = mask.clone()
#             masked[i:] = 0
#             mask_tensor = torch.cat((mask_tensor, masked))
#     mask_tensor = mask_tensor.contiguous().view(batch_size, len_sentence, -1)
#     return mask_tensor.long().to(DEVICE)


# 自定义遮罩负对数似然损失，对于超过句子长度的不计算损失
# def maskNLLLoss(output, target, mask, idx):
#     loss = torch.gather(output, 1, target)
#     mask = mask.ge(idx + 1).contiguous().view(-1, 1).float()
#     return torch.mul(loss, mask).sum().to(device)

# 自定义序列遮罩损失
# import torch
# from parameters import TGT_LENGTH, DEVICE, PAD_token
# import numpy as np


# def mask_seq_loss(decoder, decoder_output, decoder_gold, decoder_len_mask, batch_size):
#     """use a shared output project layer(linear + LogSoftmax) to project the hidden to the vocab distribution and calculate loss"""
#     cross_entrophy_loss = torch.nn.CrossEntropyLoss(ignore_index = PAD_token)
#     train_predicted = np.zeros(shape=(0, batch_size))
#     for cur in range(TGT_LENGTH):
#         # select current word position batched embedding
#         # [batch, 1(current_word_idx), hidden_size]
#         cur_output = torch.index_select(decoder_output, -2, torch.tensor(cur).to(DEVICE))

#         # project it to vocab score and squeeze the dimension
#         # [batch, vocab_size]
#         cur_vocab_score = decoder.fc(cur_output).squeeze()

#         # get model predicted output when training
#         train_predicted = np.row_stack((train_predicted, torch.max(cur_vocab_score, 1)[1].cpu().numpy()))

#         # make select mask to select batched target word onehot on current word position
#         # [batch, 1]
#         cur_select_batch = torch.tensor([cur for _ in range(batch_size)]).contiguous().view(batch_size, -1).to(DEVICE)

#         # select batched target word onehot based on select mask and squeeze it
#         # [batch]
#         cur_tgt_batch = torch.gather(decoder_gold, 1, cur_select_batch).squeeze()

#         # calculate loss per word position
#         # [batch, 1]
#         loss = cross_entrophy_loss(cur_vocab_score, cur_tgt_batch)

#         # get the length mask based on current idx and sentence length
#         # [batch, 1]
#         length_mask = decoder_len_mask.ge(cur + 1).contiguous().view(-1, 1).float()

#         # # filter the loss with filter mask
#         # # [batch, 1]
#         # if cur == 0:
#         #     loss_total = torch.mul(loss, length_mask).sum()
#         # else:
#         #     loss_total += torch.mul(loss, length_mask).sum()

#         if cur == 0:
#             loss_total = loss.sum()
#         else:
#             loss_total += loss.sum()
#     return loss_total, train_predicted.T

# 在训练迭代时打印，确定数据size正确
# print(encoder_input.size())  # [batch_size, src_len]
# print(decoder_input.size())  # [batch_size, tgt_len]
# print(decoder_len_mask.size())  # [batch_size]
# print(decoder_gold.size())  # [batch_size, tgt_len]
# print(encoder_input[0])
# print(decoder_input[0])
# print(decoder_gold[0])
# print(decoder_len_mask[0])

# 创建模型反向传播图
# g = make_graph(decoder_output)
# g.view()

# 打印模型参数
# show_param(encoder)
# show_param(decoder)

# def mask_seq_loss_eval(decoder, decoder_output, decoder_gold, decoder_len_mask, encoder_input_embed, encoder_output, decoder_input):
#     """use a shared output project layer(linear + LogSoftmax) to project the hidden to the vocab distribution and calculate loss"""
#     """ no teacher forcing """
#     cross_entrophy_loss = torch.nn.CrossEntropyLoss()
#     train_predicted = np.zeros(shape=(0, BATCH_SIZE_VALI))
#     for cur in range(TGT_LENGTH):
#         # select current word position batched embedding
#         # [batch, 1(current_word_idx), hidden_size]
#         cur_output = torch.index_select(decoder_output, -2, torch.tensor(cur).to(DEVICE))

#         # project it to vocab score and squeeze the dimension
#         # [batch, vocab_size]
#         cur_vocab_score = decoder.fc(cur_output).squeeze()

#         # current pos predict wrod
#         cur_vocab_predicted = torch.max(cur_vocab_score, 1)[1]

#         # update decoder input based on current output
#         decoder_input[:, cur + 1:cur + 2] = cur_vocab_predicted.contiguous().view(-1, 1)

#         # get model predicted output when training
#         eval_predicted = np.row_stack((train_predicted, cur_vocab_predicted.cpu().numpy()))

#         # make select mask to select batched target word onehot on current word position
#         # [batch, 1]
#         cur_select_batch = torch.tensor([cur for _ in range(BATCH_SIZE_VALI)]).contiguous().view(BATCH_SIZE_VALI, -1).to(DEVICE)

#         # select batched target word onehot based on select mask and squeeze it
#         # [batch]
#         cur_tgt_batch = torch.gather(decoder_gold, 1, cur_select_batch).squeeze()

#         # calculate loss per word position
#         # [batch, 1]
#         loss = cross_entrophy_loss(cur_vocab_score, cur_tgt_batch)

#         # get the length mask based on current idx and sentence length
#         # [batch, 1]
#         length_mask = decoder_len_mask.ge(cur + 1).contiguous().view(-1, 1).float()

#         # filter the loss with filter mask
#         # [batch, 1]
#         if cur == 0:
#             loss_total = torch.mul(loss, length_mask).sum()
#         else:
#             loss_total += torch.mul(loss, length_mask).sum()

#         decoder_output = decoder(encoder_input_embed, encoder_output, decoder_input)

#     return loss_total, eval_predicted.T

# # do attention only in last layer
# # [B, Tg, Ts]
# _attn_matrix = torch.bmm(
#     conv_output.permute(0, 2, 1), encoder_output.permute(0, 2, 1))
# attn_matrix = self.softmax(_attn_matrix)

# # [B, Tg, C]
# attn_weighted_context = torch.bmm(attn_matrix, attention_value)

# # [B, Tg, C]
# conv_output = (attn_weighted_context.permute(0, 2, 1) +
#                 conv_input) * self.scale
