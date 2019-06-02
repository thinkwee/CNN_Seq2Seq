from model import Encoder, IncrementalDecoder, Decoder, RNNDecoder
from parameters import *
import torch
import torch.utils.data as Data
import _pickle as pickle
import numpy as np

# 创建编码器解码器
encoder = Encoder().to(DEVICE)
# decoder = IncrementalDecoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
# decoder = RNNDecoder().to(DEVICE)
print("Seq2seq model built")


# 统计参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("encoder params %d" % count_parameters(encoder))
print("decoder params %d" % count_parameters(decoder))

# 加载模型用于测试
# encoder.load_state_dict(torch.load('./save_model/RNN/encoder_params_e9.pkl'))
# decoder.load_state_dict(torch.load('./save_model/RNN/decoder_params_e9.pkl'))
encoder.load_state_dict(torch.load('./save_model/encoder_params_fe.pkl'))
decoder.load_state_dict(torch.load('./save_model/decoder_params_fe.pkl'))
print("Seq2seq model loaded")

# 加载训练数据做测试
# test_src_onehot = pickle.load(open(TRAIN_DATA_PATH[0], "rb")).to(DEVICE)
# test_tgt_onehot = pickle.load(open(TRAIN_DATA_PATH[1], "rb")).to(DEVICE)
# test_len_mask = pickle.load(open(TRAIN_DATA_PATH[2], "rb")).to(DEVICE)
# test_output_onehot = pickle.load(open(TRAIN_DATA_PATH[3], "rb")).to(DEVICE)
# test_dataset = Data.TensorDataset(test_src_onehot, test_tgt_onehot, test_len_mask, test_output_onehot)
# test_loader = Data.DataLoader(dataset=test_dataset, batch_size=BATCH_SIZE_TEST, shuffle=True, drop_last=True)
# print("test data(train) loaded")

# 加载测试数据做测试
test_src_onehot = pickle.load(open(TEST_DATA_PATH[0], "rb")).to(DEVICE)
test_tgt_onehot = pickle.load(open(TEST_DATA_PATH[1], "rb")).to(DEVICE)
test_len_mask = pickle.load(open(TEST_DATA_PATH[2], "rb")).to(DEVICE)
test_output_onehot = pickle.load(open(TEST_DATA_PATH[3], "rb")).to(DEVICE)
test_dataset = Data.TensorDataset(test_src_onehot, test_tgt_onehot,
                                  test_len_mask, test_output_onehot)
test_loader = Data.DataLoader(dataset=test_dataset,
                              batch_size=BATCH_SIZE_TEST,
                              shuffle=False,
                              drop_last=True)
print("test data loaded")

# 开启训练模式
# encoder.train()
# decoder.train()

# 开启推理模式
encoder.eval()
decoder.eval()

# 加载词典
id2word = pickle.load(open("./data/id2word.dat", "rb"))

# 设置损失
cross_entrophy_loss = torch.nn.CrossEntropyLoss(ignore_index=PAD_token)

# # incremental decoding
# with torch.no_grad():
#     for step, (test_encoder_input, test_decoder_input, test_decoder_len_mask,
#             test_decoder_gold) in enumerate(test_loader):
#         # 编码
#         test_encoder_output, test_encoder_input_embed = encoder(test_encoder_input)

#         # 解码
#         loss, test_predicted = decoder(test_encoder_input_embed, test_encoder_output, test_decoder_input ,test_decoder_gold, 0.0)
#         loss /= sum(test_decoder_len_mask)

#         # 计算精确度（对于一个batch里第一份文摘）
#         accuracy_matrix = test_predicted[0] - test_decoder_gold[0].cpu().numpy()
#         accuracy = float(np.sum(accuracy_matrix[:test_decoder_len_mask[0]] == 0)) / float(test_decoder_len_mask[0])

#         for i in range(1):
#             # 每个batch检查打印第一份文摘
#             print("loss %f | accuracy %f"%(loss,accuracy))
#             print("gold\n--------------------")
#             print(' '.join([id2word[id] for id in test_decoder_gold[i].cpu().numpy()]))
#             print("output\n--------------------")
#             print(' '.join([id2word[id] for id in test_predicted[i]]))
#             print("\n")
#             s = input("pause")

#         # for i in range(BATCH_SIZE_TEST):
#         #     # 批量生成用于测试ROUGE
#         #     with open("/home/lyn/ROUGE/ROUGE-1.5.5/RELEASE-1.5.5/lw_conv_incre/systems/system." + str(step * BATCH_SIZE_TEST + i) + ".txt", "w") as f:
#         #         f.writelines(' '.join([id2word[id] for id in test_decoder_gold[i].cpu().numpy()]))
#         #     with open("/home/lyn/ROUGE/ROUGE-1.5.5/RELEASE-1.5.5/lw_conv_incre/models/model.A." + str(step * BATCH_SIZE_TEST + i) + ".txt", "w") as f:
#         #         f.writelines(' '.join([id2word[id] for id in test_predicted[i]]))
#         #     print(step * BATCH_SIZE_TEST + i)

# parallel decoding
with torch.no_grad():
    for step, (test_encoder_input, test_decoder_input, test_decoder_len_mask,
               test_decoder_gold) in enumerate(test_loader):
        # 编码
        test_encoder_output, attention_value, encoder_padding_mask = encoder(
            test_encoder_input)

        decoder_input = torch.zeros(BATCH_SIZE_TEST, TGT_LENGTH).long()
        decoder_input[:, 0] = SOS_token
        final_output = torch.zeros(BATCH_SIZE_TEST, TGT_LENGTH).long()
        final_output_score = torch.zeros(BATCH_SIZE_TEST, TGT_LENGTH,
                                         VOCAB_SIZE_TGT).to(DEVICE)

        for idx in range(TGT_LENGTH - 1):
            # 解码
            test_decoder_output, avg_attn_scores = decoder(
                attention_value, test_encoder_output, encoder_padding_mask,
                decoder_input)
            test_predicted = torch.max(test_decoder_output, 2)[1]
            decoder_input[:, idx + 1] = test_predicted[:, idx]
            final_output[:, idx] = test_predicted[:, idx]
            final_output_score[:, idx, :] = test_decoder_output[:, idx, :]

        loss = cross_entrophy_loss(
            final_output_score.contiguous().view(-1,
                                                 final_output_score.shape[-1]),
            test_decoder_gold.contiguous().view(-1))

        # 计算精确度（对于一个batch里第一份文摘）
        accuracy_matrix = final_output[0].numpy() - test_decoder_gold[0].cpu(
        ).numpy()
        accuracy = float(
            np.sum(accuracy_matrix[:test_decoder_len_mask[0]] == 0)) / float(
                test_decoder_len_mask[0])

        for i in range(1):
            # 每个batch检查打印第一份文摘
            print("loss %f | accuracy %f" % (loss, accuracy))
            print("original\n--------------------")
            print(' '.join(
                [id2word[id] for id in test_encoder_input[i].cpu().numpy()]))
            print("gold\n--------------------")
            print(' '.join(
                [id2word[id] for id in test_decoder_gold[i].cpu().numpy()]))
            print("output\n--------------------")
            print(' '.join([id2word[id] for id in final_output[i].numpy()]))
            print("\n")
            s = input("pause")

        # 批量生成用于测试ROUGE
        # for i in range(BATCH_SIZE_TEST):
        #     with open("/home/lyn/ROUGE/ROUGE-1.5.5/RELEASE-1.5.5/lw_conv/systems/system." + str(step * BATCH_SIZE_TEST + i) + ".txt", "w") as f:
        #         f.writelines(' '.join([id2word[id] for id in test_decoder_gold[i].cpu().numpy()]))
        #     with open("/home/lyn/ROUGE/ROUGE-1.5.5/RELEASE-1.5.5/lw_conv/models_witheos/model.B." + str(step * BATCH_SIZE_TEST + i) + ".txt", "w") as f:
        #         f.writelines(' '.join([id2word[id] for id in final_output[i].numpy()]))
        #     print(step * BATCH_SIZE_TEST + i)