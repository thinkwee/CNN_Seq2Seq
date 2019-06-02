from model import Encoder, Decoder
from parameters import *
import torch
import torch.utils.data as Data
import _pickle as pickle
from utils import make_graph, show_param
import time
from visdom import Visdom
import numpy as np

# 记录损失
viz = Visdom(env=u'fast convergence')
opts_loss = {
    'title': 'sequence loss',
    'xlabel': 'every 200 batch',
    'ylabel': 'loss',
    'showlegend': 'true'
}
opts_acc = {
    'title': 'Accuracy',
    'xlabel': 'every 200 batch',
    'ylabel': 'accuracy',
    'showlegend': 'true'
}

# 创建编码器解码器
encoder = Encoder().to(DEVICE)
decoder = Decoder().to(DEVICE)
viz.text("Seq2seq model built", win='summary')


# 统计参数量
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print("encoder params %d" % count_parameters(encoder))
print("decoder params %d" % count_parameters(decoder))

# 加载预训练词嵌入
encoder.load_pretrained_vectors(EMBED_PATH_SRC)
decoder.load_pretrained_vectors(EMBED_PATH_TGT)
viz.text("Pretrained embeddings loaded", win='summary', append=True)

# 创建优化器
enc_optimizer = torch.optim.Adam(encoder.parameters())
dec_optimizer = torch.optim.Adam(decoder.parameters())
viz.text("Optimizer created", win='summary', append=True)

# 加载训练数据做训练
train_src_onehot = pickle.load(open(TRAIN_DATA_PATH[0], "rb")).to(DEVICE)
train_tgt_onehot = pickle.load(open(TRAIN_DATA_PATH[1], "rb")).to(DEVICE)
train_len_mask = pickle.load(open(TRAIN_DATA_PATH[2], "rb")).to(DEVICE)
train_output_onehot = pickle.load(open(TRAIN_DATA_PATH[3], "rb")).to(DEVICE)
viz.text("train data loaded", win='summary', append=True)

# 加载测试数据做训练
# train_src_onehot = pickle.load(open(TEST_DATA_PATH[0], "rb")).to(DEVICE)
# train_tgt_onehot = pickle.load(open(TEST_DATA_PATH[1], "rb")).to(DEVICE)
# train_len_mask = pickle.load(open(TEST_DATA_PATH[2], "rb")).to(DEVICE)
# train_output_onehot = pickle.load(open(TEST_DATA_PATH[3], "rb")).to(DEVICE)
# viz.text("train(test) data loaded", win='summary', append=True)

# 创建data loader
train_dataset = Data.TensorDataset(train_src_onehot, train_tgt_onehot,
                                   train_len_mask, train_output_onehot)
train_loader = Data.DataLoader(dataset=train_dataset,
                               batch_size=BATCH_SIZE_FAST,
                               shuffle=True,
                               drop_last=True)
viz.text("data loader created", win='summary', append=True)

# 开启训练模式
encoder.train()
decoder.train()

# 记录开始时间
start = time.process_time()
viz.text(time.strftime("STARTS AT %a %b %d %H:%M:%S %Y \n", time.localtime()),
         win='summary',
         append=True)

# step for counting loss
step_global = 0

# 加载词典
# id2word = pickle.load(open("./data/id2word.dat", "rb"))

# 设置损失
cross_entrophy_loss = torch.nn.CrossEntropyLoss(ignore_index=PAD_token)

for epoch in range(EPOCH_FAST_CONVERGENCE):
    for step, (train_encoder_input, train_decoder_input,
               train_decoder_len_mask,
               train_decoder_gold) in enumerate(train_loader):
        update = 'append' if step_global > 1 else None

        # 清空上一次bp的梯度
        enc_optimizer.zero_grad()
        dec_optimizer.zero_grad()

        # 编码
        train_encoder_output, attention_value, encoder_padding_mask = encoder(
            train_encoder_input)

        # 解码
        train_decoder_output, avg_attn_scores = decoder(
            attention_value, train_encoder_output, encoder_padding_mask,
            train_decoder_input)

        loss = cross_entrophy_loss(
            train_decoder_output.contiguous().view(
                -1, train_decoder_output.shape[-1]),
            train_decoder_gold.contiguous().view(-1))

        train_predicted = torch.max(train_decoder_output, 2)[1].cpu().numpy()

        # 计算梯度
        loss.backward()

        # 梯度截断
        _ = torch.nn.utils.clip_grad_norm_(encoder.parameters(), CLIP)
        _ = torch.nn.utils.clip_grad_norm_(decoder.parameters(), CLIP)

        # 参数更新
        enc_optimizer.step()
        dec_optimizer.step()

        # 增加loss图横坐标
        step_global += 1

        # for i in range(1):
        #     # 每个batch打印一次，检查文摘
        #     print("fast convergence loss")
        #     print(loss)
        #     print("gold\n--------------------")
        #     print(' '.join([id2word[id] for id in train_decoder_gold[i].cpu().numpy()]))
        #     print("output\n--------------------")
        #     print(' '.join([id2word[id] for id in train_predicted[i]]))
        #     print("\n")
        #     s = input("pause")

        # 每隔PRINT_EVERY个batch记录训练损失和精度
        if step % PRINT_EVERY_SMALL == 0:
            # 打印并记录损失
            viz.text(str('epoch: {}    batch: {}   loss: {:.4f}  \n'.format(
                epoch, step, loss.data)),
                     win='summary',
                     append=True)

            # 更新accuracy图
            accuracy_matrix = train_predicted - train_decoder_gold.cpu().numpy(
            )
            accuracy = 0.0
            for i in range(BATCH_SIZE_FAST):
                accuracy += float(
                    np.sum(
                        accuracy_matrix[i][:train_decoder_len_mask[i]] == 0))
            accuracy /= float(sum(train_decoder_len_mask))
            viz.line(X=torch.FloatTensor([step_global]),
                     Y=torch.FloatTensor([accuracy]),
                     win='acc',
                     update=update,
                     opts=opts_acc)

            # 更新loss图
            viz.line(X=torch.FloatTensor([step_global]),
                     Y=torch.FloatTensor([loss]),
                     win='loss',
                     update=update,
                     opts=opts_loss)

    # 每个epoch结束时保存一次模型，覆盖上一次保存，最后只留下最后一次迭代结果
    torch.save(encoder.state_dict(), './save_model/encoder_params_fe.pkl')
    torch.save(decoder.state_dict(), './save_model/decoder_params_fe.pkl')

    # 记录结束时间
    end = time.process_time()
    viz.text(time.strftime("ENDS AT %a %b %d %H:%M:%S %Y \n\n",
                           time.localtime()),
             win='summary',
             append=True)
    m, s = divmod(end - start, 60)
    h, m = divmod(m, 60)
    viz.text("time cost: %02d:%02d:%02d" % (h, m, s),
             win='summary',
             append=True)
