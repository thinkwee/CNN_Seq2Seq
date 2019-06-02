# CNN_Seq2Seq

-   卷积端到端的最简实现，可用于实现文档级生成式文摘（然而还是做翻译效果更好）

# TODO
-   [x] CNN Encoder
-   [x] CNN Decoder
-   [x] Multi-step Attention
-   [x] Dilation
-   [x] Output all logs to Visdom
-   [ ] Transform CNN to FC in Decoder when infer
-   [ ] Adaptive Softmax
-   [ ] Efficient memory fp-16
-   [ ] For other tasks

# 环境
-   python 3.7
-   ubuntu 16.04
-   pytorch 1.0
-   Visdom
-   如果需要预训练词嵌入需自己配置fasttext

# 获取数据
-   创建data文件夹，下载[CNNDM数据集](https://drive.google.com/open?id=1buWz_W4slL2GPt4EPYQI7Lf0kkHfAtLT)

# 预处理
-   运行preprocess.ipynb

# 超参调整
-   见parameters.py

# 训练
-   python train.py

# 测试
-   python infer.py

# notebook处理数据
-	data_presentation.ipynb：数据集统计
-	make_pretrained_embedding.ipynb：从fasttext预训练好的词嵌入中挑出模型词典构成嵌入矩阵
-	preprocess.ipynb：对cnndm数据集的预处理
-	tensor_test.ipynb：其他测试

# python模型训练测试
-	conv_seq2seq.py：卷积端到端模型，包括编码器解码器两个类
-	deprecated_code.py：一些废弃的代码
-	infer.py:模型推理
-	layers.py：自定义权重初始化的全连接层、卷积层、遮罩了的时序卷积层
-	loss.py：对decoder出来的每一个时间步进行词典范围投影，并计算序列的交叉熵损失，带遮罩
-	paramcount.py：统计模型参数量
-	parameters.py：模型的所有超参数
-	train.py：模型训练
-	visualization.py：模型计算图可视化

# 临时创建文件夹
-   model_check：监视模型训练情况，包括训练log，记录的损失以及训练输出
-   model_graph：调用可视化得到的模型反向传播计算图
-   save_model：保存的模型
-   model_output、system_output：得到文摘用于计算ROUGE

# 效果
![figure](https://github.com/thinkwee/CNN_Seq2Seq/blob/master/sample.png)