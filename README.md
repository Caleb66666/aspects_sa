## 细粒度分类
多标签，多分类问题

### 基础框架
1. 使用albert_zh的分词器，即以词片作为词嵌入，并允许词嵌入fine-tune，其他参数固定，解决oov问题
2. 共享层：使用encoder作为语句序列表征
3. 独立层：针对每个标签，均单独定义一个基于attention-pooling的分类器
4. 损失：基础的交叉熵，每个单独分类器的损失的mean，且每个损失的权重均等为1
5. 精度：单独计算每个分类器的f1(micro), 计算所有f1的mean，每个f1的权重均等于1
6. 优化器：AdamW
7. 学习率调节器：warm-up，三角学习率

### base-line模型参数设置
- epoch: 30
- encoder: 常规lstm，encoder-size 256
- lr: 6e-5
- warm_up_proportion: 0.1 
- max-seq: 1024 基本覆盖98.5%的语句不需要截断，即使极端，也使用前后截法
- linear_size: 每个分类器的线性大小，128
- embed_dim: 使用的albert_zh是base版本，其词嵌入大小为128
- max_grad_norm：梯度修剪的幅度

### 效果
以最终损失衡量模型效果best-valid-loss，每个版本均从base-line上进行改动

#### base-line
基础版本的valid-loss: 0.3866
选择albert是因为其词表征输出维度较小，而且本身该模型为蒸馏模型，训练步骤及其长，结合其基于sub word的分词方法。不仅可以比较完美的解决oov问题，而且还拥有维度小，表征能力强的词嵌入，结果证明，后续接入一个比较简单的双向LSTM作为序列表征，分类器使用attention+max_pool就能获得一个较好的基线结果。

#### 改进一
考虑更大的规模和size，不过效果提升比较少: 0.37212
- 更大的lstm_size:512
- num_layers:2
- 更大的linear_size: 256
- 更大的attention：attn_size: 128，初始化为Parameters(), 在`tanh`激活之前与输入tensor相乘
- 更大的pool: 由max-pool -> cat([avg_pool, max_pool])

#### 改进二：
考虑改进序列表征，引入简化版的elmo
