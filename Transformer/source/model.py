import math
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib.pyplot as plt
from data_tool import *

# 模型超参数
embedding_dim = 512  # 嵌入维度
n_heads = 8  # 多头注意力头数
n_layers = 6  # 编码器和解码器层数

class Transformer(nn.Module):
    """一个标准的transformer模型

    包括了嵌入、位置编码、编码器层、编码器、解码器层、解码器等模块
    向前传播步骤：
        1. 对源序列和目标序列进行嵌入操作, 嵌入维度512
        2. 增加位置编码
        3. 丢弃部分数据，防止过拟合
        4. 编码
        5. 解码
        6. 计算输出
    """
    def __init__(self, vocab_size, embedding_dim, n_heads, n_layers, max_seq_length, dropout=0.1):
        super(Transformer, self).__init__() # 继承父类nn.Module
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")  # 获取设备
        self.embedding = nn.Embedding(vocab_size, embedding_dim).to(self.device)  # 嵌入矩阵
        self.positional_encoding = self.create_positional_encoding(embedding_dim, max_seq_length).to(self.device)  # 位置编码
        encoder_layer = nn.TransformerEncoderLayer(embedding_dim, n_heads, dim_feedforward=2048, dropout=dropout, batch_first=True).to(self.device)  # 编码层
        self.encoder = nn.TransformerEncoder(encoder_layer, n_layers).to(self.device)  # 编码器
        decoder_layer = nn.TransformerDecoderLayer(embedding_dim, n_heads, dim_feedforward=2048, dropout=dropout, batch_first=True).to(self.device)  # 解码层
        self.decoder = nn.TransformerDecoder(decoder_layer, n_layers).to(self.device)  # 解码器
        self.fc_out = nn.Linear(embedding_dim, vocab_size).to(self.device)  # 输出层
        self.dropout = nn.Dropout(dropout).to(self.device)  # 丢弃

    def create_positional_encoding(self, embedding_dim, max_seq_length):
        positional_encoding = torch.zeros(max_seq_length, embedding_dim)    # 初始化位置编码矩阵
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)      # 位置信息 计算公式分子
        div_term = torch.exp(torch.arange(0, embedding_dim, 2).float() * (-math.log(10000.0) / embedding_dim))  # 计算公式分母
        positional_encoding[:, 0::2] = torch.sin(position * div_term)
        positional_encoding[:, 1::2] = torch.cos(position * div_term)
        return positional_encoding
    
    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        # 源序列、目标序列嵌入
        src_embed = self.embedding(src)     # [batch_size, max_seq_length, embedding_dim]
        tgt_embed = self.embedding(tgt)     # [batch_size, max_seq_length, embedding_dim]
        # 生成位置编码
        batch_size = src.size(0)
        posit = self.positional_encoding.unsqueeze(0).expand(batch_size, -1, -1)   # 扩展至批次维度[batch_size, max_seq_length, embedding_dim]
        src_temp = src_embed + posit
        tgt_temp = tgt_embed + posit
        # 丢弃部分数据，避免过拟合
        src_temp = self.dropout(src_temp)
        tgt_temp = self.dropout(tgt_temp)
        memory = self.encoder(src_temp, src_mask)  # 编码
        output = self.decoder(tgt_temp, memory, tgt_mask, memory_mask) # 解码
        output = self.fc_out(output)    # 计算输出
        return output

    def generate(self, src, beam_size=5, early_stopping=True):
        # 将输入数据转换为模型可以使用的格式
        src = src.to(self.device)
        with torch.no_grad():
            # 首先进行编码
            encoder_output = self.encoder(self.embedding(src) * math.sqrt(embedding_dim))
            # 初始化解码器输入（通常为起始符号）
            start_symbol_index = torch.tensor([1], dtype=torch.long).to(self.device)
            start_symbol_embedding = self.embedding(start_symbol_index)
            # 初始化输出序列
            output_sequence = start_symbol_embedding
            # 初始化解码器输入
            decoder_input = start_symbol_embedding.long()
            # 计算嵌入向量
            embedded_input = self.embedding(decoder_input)
            # 应用嵌入向量的缩放
            scaled_embedded_input = embedded_input * math.sqrt(embedding_dim)
            
            tgt_mask = torch.tril(torch.ones(embedding_dim, embedding_dim,dtype=bool), diagonal=0).unsqueeze(0).expand(len(src)*n_heads, -1, -1).to(self.device)

            # 解码过程
            for _ in range(max_seq_length):
                decoder_output = self.decoder(scaled_embedded_input, encoder_output, tgt_mask)
                decoder_output = self.fc_out(decoder_output)
                decoder_output = F.log_softmax(decoder_output, dim=2)

                # 选择最可能的下一个词
                if beam_size > 1:
                    # 使用beam search
                    topk_scores, topk_indices = torch.topk(decoder_output.squeeze(0), beam_size)
                    topk_indices = topk_indices.unsqueeze(0)
                    decoder_input = torch.cat([decoder_input, self.embedding(topk_indices)], dim=0)
                    output_sequence = torch.cat([output_sequence, self.embedding(topk_indices)], dim=0)
                else:
                    # 直接采样
                    _, next_word = decoder_output.max(dim=2)
                    next_word_embedding = self.embedding(next_word)
                    decoder_input = next_word_embedding
                    output_sequence = torch.cat([output_sequence, next_word_embedding], dim=0)

                # 如果使用了early stopping，可以提前结束生成
                if early_stopping and torch.any(next_word == 0):
                    break

            # 删除额外的维度
            output_sequence = output_sequence.squeeze(0)

            return output_sequence
               
def shift_right(tensor, pad_value=0):
    """Decoder输入tgt右移处理"""
    # 获取张量的形状
    batch_size, seq_length = tensor.size()
    # 创建一个新的张量，填充 pad_value
    shifted_tensor = torch.full((batch_size, seq_length), pad_value, dtype=tensor.dtype)
    # 将原始张量的内容复制到新的张量中，向右移动一位
    shifted_tensor[:, 1:] = tensor[:, :-1]
    return shifted_tensor

def get_path(type,test):
    if type == 0 and test == True:
        path = r"model_save\small\English_to_Chinese_model.pth"
    elif type == 0 and test == False:
        path = r"model_save\big\English_to_Chinese_model.pth"
    elif type == 1 and test == True:
        path = r"model_save\small\Chinese_to_English_model.pth"
    elif type == 1 and test == False:
        path = r"model_save\big\Chinese_to_English_model.pth"
    return path

def save_model(model, type=0, test=False):
    """保存模型"""
    save_path = get_path(type,test)
    # 保存模型
    torch.save(model.state_dict(), save_path)
    print("已保存模型至",save_path)

def load_model(type, test=False):
    """加载模型"""
    load_path = get_path(type,test)
    # 加载模型
    model = Transformer(vocab_size, embedding_dim, n_heads, n_layers, max_seq_length)
    model = model.to(model.device)
    model.load_state_dict(torch.load(load_path))
    return model

def get_src_tgt(English, Chinese, model, type=0):
    if type==0:
        src, tgt = English, Chinese
    else:
        src, tgt = Chinese, English
    shifted_right_tgt = shift_right(tgt)    # tgt右移
    src = src.to(model.device)    # 放入GPU
    shifted_right_tgt = shifted_right_tgt.to(model.device)    # 放入GPU
    return src,shifted_right_tgt

def create_tgt_mask(tgt,model):
    length = len(tgt)
    tgt_mask = torch.triu(torch.ones(max_seq_length, max_seq_length,dtype=bool), diagonal=1).unsqueeze(0).expand(length*n_heads, -1, -1).to(model.device)
    return tgt_mask
def create_src_mask(src,model):

    return
def create_memory_mask(src,model):

    return 

def draw_loss(loss_values):
# 绘制平均损失曲线图
    plt.figure(figsize=(10, 5))
    plt.plot(loss_values, label='Average Loss Every 10 Batchs')
    plt.title('Average Loss Over Batchs Every 10 Batchs')
    plt.xlabel('Batch (Every 10 Batch)')
    plt.ylabel('Average Loss')
    plt.legend()
    plt.grid(True)
    plt.show()

def model_train(train_loader, val_loader, type=0, num_epochs=3, test=False):
    """训练模型
    Args:
        data_loader:训练数据加载器
        type: 训练类型, 0——English to Chinese, 1——Chinese to English
        epochs: 训练轮数
        test: 是否为测试
    UserWarning: Torch was not compiled with flash attention.
    FlashAttention only supports Ampere GPUs or newer. 至少RTX 3060才能跑得起来。
    本机器暗影精灵7 RTX 3050, 硬件不支持

    1. 训练步骤
    2. 早停机制：连续 patience 个 epoch 验证集损失没有下降就停止训练
    3. 损失曲线绘制

    """
    total_training_time = 0 # 初始化总训练时间
    loss_values = []        # 损失列表
    validation_frequency = 1000 # 模型验证周期
    patience = 5  # 如果连续5次验证集损失没有改善，则停止
    best_val_loss = float('inf')  # 初始化最佳验证损失
    patience_counter = 0  # 早停计数器
    
    if test==True:
        num_epochs = 10
        validation_frequency = 30  
    
    # 1. 创建一个Transformer模型
    model = Transformer(vocab_size, embedding_dim, n_heads, n_layers, max_seq_length)
    model = model.to(model.device)    # 放入GPU
    # 2. 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
    # 3. 定义学习率调度器
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=3, threshold=0.0001, cooldown=1, min_lr=1e-8)
    # 4. 循环训练    
    for epoch in range(num_epochs):
        num_batches = 0    # 记录批次数
        start_time = time.time()  # 记录训练开始时间
        print(f"epoch [{epoch+1}/{num_epochs}]")
        batch_loss_10 = 0.0
        for English, Chinese in train_loader:
            num_batches += 1    # 记录批次
            src, tgt = get_src_tgt(English,Chinese,model,type)
            tgt_mask = create_tgt_mask(tgt,model)
            # 前向传播、计算损失、反向传播
            optimizer.zero_grad()
            output = model(src, tgt, tgt_mask)
            loss = criterion(output.view(-1, vocab_size), tgt.view(-1))
            loss.backward()
            optimizer.step()

            batch_loss_10 += loss.item()
            if num_batches % 10 == 0:
                # 训练时间、损失记录和日志输出
                average_loss = batch_loss_10 / 10
                loss_values.append(average_loss)
                end_time = time.time()
                training_time = end_time - start_time
                total_training_time += training_time
                print(f"[Batch {num_batches}], Training Time: {training_time:.2f} seconds")
                print(f"[Batch {num_batches}], Average Loss : {average_loss:.4f}")
                start_time = time.time()
                batch_loss_10 = 0.0

            if num_batches % validation_frequency == 0:
                # 验证、早停
                model.eval()
                with torch.no_grad():
                    num_val_batchs = 0
                    val_ave_loss = 0.0
                    for English, Chinese in val_loader:
                        num_val_batchs += 1
                        val_src, val_tgt = get_src_tgt(English, Chinese, model, type)
                        tgt_mask = create_tgt_mask(tgt,model)
                        val_output = model(val_src,val_tgt, tgt_mask)
                        val_loss = criterion(val_output.view(-1, vocab_size), val_tgt.view(-1))
                        val_ave_loss += val_loss.item() / len(val_src)

                    # 保存最佳验证损失和模型
                    if val_ave_loss < best_val_loss:
                        best_val_loss = val_ave_loss
                        patience_counter = 0  # 重置计数器
                    else:
                        patience_counter += 1
                    # 打印当前批次的损失
                    print("This is a Valuation: ")
                    print(f"Last Batch Loss: {loss}")
                    print(f"Validation Loss: {val_loss}")   
                    # 检查是否满足早停条件
                    if patience_counter >= patience:
                        print("Early stopping!")
                        break
                    else:
                        print("Continue!")
 
                # 更新学习率
                scheduler.step(loss.item())

    print(f"The Last Loss: {loss}")
    print(f"Total Training Time: {total_training_time:.2f} seconds")
    # 损失函数图
    draw_loss(loss_values)
    # 保存模型
    save_model(model,type,test)

def predict(src_text, type=0, test=False):
    # 加载模型
    transformer = load_model(type,test)
    # 将源文本转换为模型所需的格式
    if type==0: src_text = src_text.lower()
    src_encoded = tokenizer(src_text, max_length=max_seq_length, truncation=True, padding='max_length', return_tensors='pt').to(transformer.device)
    # 使用模型进行预测
    with torch.no_grad():  # 关闭梯度计算
        outputs = transformer.generate(src_encoded['input_ids'])
    # 解码预测结果
    pred_text = output_to_seq(outputs)
    return pred_text

def train(train_data_path, val_data_path, test=False):
    print("----------模型训练测试----------")
    train_loader, train_dataset = data_loader(train_data_path)
    val_loader, val_dataset = data_loader(val_data_path)
    model_train(train_loader,val_loader,test=test)

if __name__=="__main__":
    train_data_path = r"data\small\train.txt"
    val_data_path = r"data\small\val.txt"
    train(train_data_path,val_data_path,test=True)
    