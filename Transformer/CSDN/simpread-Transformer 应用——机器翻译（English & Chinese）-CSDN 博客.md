> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/T940842933/article/details/140738974?spm=1001.2014.3001.5502)

说明
--

本片博客记录在学习论文《Attention is all you need》和项目《tensor2tensor》的基础上进行的实际应用情况。不做任何商业用途，绝不允许侵权行为。

论文地址：[Attention is all you need](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf "Attention is all you need")

项目地址：[tensor2tensor](https://github.com/tensorflow/tensor2tensor "tensor2tensor")

本项目源码：[github.com](https://github.com/hunnyTH/ChatGPT_Learning/tree/main/Transformer "github.com")

学习工具：Visual Studio Code、ChatGPT-4o-mini、GitHub

学习内容：根据对论文和项目的理解，建立一个基础的 transformer 模型，使用中英文平行语料对其进行训练、优化、评估。

一、机器翻译
------

机器翻译模型是人工智能领域中的一种技术，用于自动将文本从一种语言翻译成另一种语言。

使用 Transformer 模型架构：

*   **序列到序列（Seq2Seq）模型**：这是机器翻译中最常见的架构，由编码器（Encoder）和解码器（Decoder）组成。编码器将输入文本编码成一个固定长度的向量，解码器则根据这个向量生成目标语言的翻译。
*   **注意力机制（Attention）**：在 Seq2Seq 模型中加入注意力机制，可以使得解码器在生成每个目标单词时，能够关注到输入序列中的多个单词，从而提高翻译质量。

训练数据流：

1.  输入模型前：将批量源序列和目标序列对 token 化并通过索引编码，使用 pad_value=0 将其补全至 max_seq_length 长度，矩阵形状均为 [batch_size, max_seq_length]
2.  在模型中：通过 embedding、positional encode、encode、decode、fc_out 等层，输出概率矩阵 [batch_size, max_seq_length, vocab_size]
3.  序列化：每一个序列可以表示为矩阵 [max_seq_length, vocab_size]，通过贪婪搜索策略或者最大概率策略获得序列 token 索引[1，max_seq_length]，则批序列索引矩阵[batch_size, max_seq_length] 进行解码获得输出字符串序列。

这是我对 transformer 模型数据流的简单理解，在下文会对相关的内容进行详细解释。

二、数据
----

 **数据来源：**[5. 翻译语料 (translation2019zh)，520 万个中英文句子对](https://github.com/brightmart/nlp_chinese_corpus "5.翻译语料(translation2019zh)，520万个中英文句子对")

**直接下载：**[https://drive.google.com/file/d/1EX8eE5YWBxCaohBO8Fh4e2j3b9C2bTVQ/view](https://drive.google.com/file/d/1EX8eE5YWBxCaohBO8Fh4e2j3b9C2bTVQ/view "https://drive.google.com/file/d/1EX8eE5YWBxCaohBO8Fh4e2j3b9C2bTVQ/view")

**数据描述：**中英文平行语料 520 万对，包括训练集 516 万、验证集 3.9 万，数据去重。

每一个对，包含一个英文和对应的中文。中文或英文，多数情况是一句带标点符号的完整的话。对于一个平行的中英文对，中文平均有 36 个字，英文平均有 19 个单词 (单词如 “she”)。例：

> {"english": "And the light breeze moves me to caress her long ear", "chinese": "微风推着我去爱抚它的长耳朵"}

### **2.1 数据集划分**

在调试过程中，训练 320 条数据平均需要 30 秒。而训练一个中英翻译器大约需要数百万条数据，若我使用完整的训练集，那么即使 epoch=1，也至少需要 130 个小时。目前我做这个项目是为了增强对 transformer 模型的理解，以及提高代码能力，并不是为了得到一个可以实用的翻译器。所以我在项目中建立了 small 和 big 文件夹，均包含训练、验证、测试数据。本次均使用 small 数据集训练，若今后有更好的硬件条件以及对代码优化完成，我会尝试训练完整的数据集。

**small：**

*   train_txt：从原训练集中提取了前 960 条数据
*   val_txt：原训练集的 [961,1280] 条数据
*   test_txt：原训练集的 [1281,1600] 条数据

**big：**

*   **train.txt（训练）：**原训练集 5161434 条数据
*   **val.txt（验证）：**原验证集前 3200 条数据
*   **test.txt（测试）：**将原验证集的剩余数据（36123 条）

注意到，数据集样本数大多都是 32 的倍数，因为我设置 **batch_size=32****，**模型采用并行式训练，一次可以将 32 条数据传入模型，这是 Transformer 架构的一个重要优点。

### 2.2 数据处理

数据集划分好后，进行处理工作，全部用函数进行封装，在 data_tool.py 中

```
import re
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
from model import vocab_size,max_seq_length,batch_size
 
def read_file(path)    # 读取文件
class TranslationDataset(Dataset)    # 自定义数据集
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')    # tonken工具
def data_loader(data_path)    # 数据加载器
def capitalize_first_letter_of_sentences(text)    # 英语句子首字母大写
def tokens_to_sequences(predicted_tokens)    # 将token处理成字符串序列
def output_to_seq(output=None, tgt_indices=None, test=False)    # 将模型输出处理为字符串序列
def test()    # 测试函数
```

 通过这些工具，我们可以完成模型输入前、输出后的绝大部分数据处理工作，主要包括

*   **读取数据：**从文件读取，并存入数据类中，并且做了填充处理；可单独访问中英文数据，用于评估时使用
*   **模型输入：**根据数据类创建了加载器，可以加载批量数据用于模型调试、训练和评估
*   **模型输出：**对于训练后得到概率矩阵，可以生成字符串序列

#### 2.2.1 读取数据

原数据集为 json 文件，每一行为一个 json 对象，读取时会报错，所以直接转为了 txt 文件按行读取，返回字典列表。

```
def read_file(path):
    """数据文件格式为txt,每一行为一个json文件, 按行读取并添加至data"""
    data = []
    try:
        # 打开文件
        with open(path, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除可能的空行或额外字符
                clean_line = line.strip()
                if clean_line:
                    try:
                        # 将字符串转换为字典
                        data_dict = json.loads(clean_line)
                        data.append(data_dict)
                    except json.JSONDecodeError as e:
                        print(f"JSON解析错误: {e}")
    except FileNotFoundError as e:
        print(f"文件未找到: {e}")
    except IOError as e:
        print(f"文件读取错误: {e}")
    return data
```

**函数说明**

*   **函数名称**：`read_file`
*   **参数**：
    *   `path`：一个字符串，表示要读取的文件的路径。
*   **返回值**：
    *   `data`：一个列表，包含从文件中解析出来的字典。

**函数功能**

1.  **打开文件**：使用`with open(path, 'r', encoding='utf-8') as file:`语句打开文件，并指定以只读模式打开，使用 UTF-8 编码。
2.  **逐行读取**：通过`for line in file:`循环逐行读取文件内容。
3.  **去除空行**：对每一行使用`line.strip()`去除可能的空白字符。
4.  **解析 JSON**：如果行内容非空，则尝试使用`json.loads(clean_line)`将字符串解析为字典。
5.  **错误处理**：
    
    1.  如果解析 JSON 时出错（例如，如果字符串不是有效的 JSON 格式），则捕获`json.JSONDecodeError`异常，并打印错误信息。
    2.  如果文件打开或读取时出错，则捕获`FileNotFoundError`或`IOError`异常，并打印错误信息。
6.  **返回数据**：函数最后返回包含所有解析字典的列表。

#### 2.2.2 数据类定义

```
class TranslationDataset(Dataset):
    """自定义数据集"""
    def __init__(self, data, tokenizer, max_length=max_seq_length):
        self.data = data    # 数据
        self.English = [item['english'].lower() for item in data]  # 将英文文本添加到 self.English 列表，编码需要小写化
        self.Chinese = [item['chinese'] for item in data]  # 将中文文本添加到 self.Chinese 列表
        self.tokenizer = tokenizer  # token化工具
        self.max_length = max_length    # 最大序列长度
 
    def __len__(self):
        return len(self.data)
 
    def __getitem__(self, idx):
        English_encoded = self.tokenizer(self.English[idx], max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        Chinese_encoded = self.tokenizer(self.Chinese[idx], max_length=self.max_length, truncation=True, padding='max_length', return_tensors='pt')
        return English_encoded['input_ids'].reshape(-1), Chinese_encoded['input_ids'].reshape(-1)
```

**类定义**

*   **类名**：`TranslationDataset`
*   **继承**：`Dataset`

**类属性**

*   `data`：包含所有数据项的列表，每个数据项是一个包含英文和中文文本的字典。
*   `English`：包含所有英文文本的列表，所有文本被转换为小写。
*   `Chinese`：包含所有中文文本的列表。
*   `tokenizer`：用于将文本转换为机器可读的 token 序列的工具。
*   `max_length`：序列的最大长度，超过这个长度的序列将被截断。

**方法**

1.  `__init__`：构造函数，初始化数据集。
2.  `__len__`：返回数据集中的数据项数量。
3.  `__getitem__`：根据索引返回一个数据项。

**构造函数 `__init__`**

*   **参数**：
    *   `data`：数据集，是一个列表。
    *   `tokenizer`：用于文本编码的 tokenizer 对象。
    *   `max_length`：可选参数，默认为`max_seq_length`，表示序列的最大长度。

构造函数中，首先将英文文本转换为小写并存储在`self.English`列表中，中文文本则直接存储在`self.Chinese`列表中。同时，将 tokenizer 和最大序列长度保存为类的属性。

**`__len__` 方法**

这个方法返回数据集中数据项的总数。

**`__getitem__` 方法**

*   **参数**：
    *   `idx`：索引，用于获取数据集中的特定项。

此方法使用提供的索引`idx`来获取英文和中文文本，然后使用 tokenizer 将它们编码为 token 序列。编码时，如果文本长度超过`max_length`，则会进行截断；如果文本长度不足，则会进行填充以达到`max_length`。编码后的结果包括`input_ids`，这是模型需要的输入格式。最后，返回两个编码后的`input_ids`，并且将它们重塑为二维数组（如果需要的话）。

#### 2.2.3 数据加载器

```
tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')
def data_loader(data_path):
    # 从路径读取数据 
    data = read_file(data_path)
    # 创建测试数据集
    dataset = TranslationDataset(data, tokenizer)
    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset
```

*   **`tokenizer`：**使用 Hugging Face 的 transformers 库加载预训练的`bert-base-chinese`模型的 tokenizer。这个 tokenizer 将用于将文本转换为模型能够理解的 token 序列。
    
*   **`data_loader`函数：**这个函数接收一个数据列表`data`，并返回一个数据加载器`loader`和一个数据集对象`dataset`。
    
    *   **参数**：
        
        *   `data`：一个列表，包含用于训练的数据项，每个数据项通常是一个包含英文和中文文本的字典。
    *   **返回值**：
        
        *   `loader`：数据加载器，用于在训练过程中迭代数据。
        *   `dataset`：数据集对象，它是`TranslationDataset`的一个实例。
    *   **函数内部操作**：
        
        *   根据传入的路径读取文件获得数据列表`data`
        *   使用 data 和 tokenizer 创建一个`TranslationDataset`实例。
        *   使用这个数据集实例创建一个`DataLoader`实例。`DataLoader`的`batch_size`参数定义了每个批次的样本数量，`shuffle`参数设置为`True`表示在每个 epoch 开始时打乱数据。

注意：

"tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')" 可能会请求失败，再次运行即可

![](https://i-blog.csdnimg.cn/direct/a7fbbb4bc12d4acd8249c71df9c454b1.png)

#### 2.2.4 序列化

```
def capitalize_first_letter_of_sentences(text):
    # 使用split方法将文本分割成句子列表，这里假设句子以点号、问号或感叹号结尾
    sentences = text.split('. ')
    sentences = [sentence.strip() for sentence in sentences if sentence.strip() != '']
    
    # 对每个句子进行处理，使其第一个单词的首字母大写
    capitalized_sentences = [sentence[0].upper() + sentence[1:] if sentence else '' for sentence in sentences]
    
    # 将处理后的句子列表重新组合成一个文本
    capitalized_text = '. '.join(capitalized_sentences) + ('.' if text.endswith('.') else '')
    
    return capitalized_text
 
def tokens_to_sequences(predicted_tokens):
    # 移除 [CLS]、[SEP]、[UNK]、[PAD]、[MASK] 标志
    processed_sentence = predicted_tokens.replace("[CLS]", "").replace("[SEP]", "").replace("[PAD]", "").replace("[UNK]", "").replace("[MASK]", "")
    # 去除多余的空格
    processed_sentence = " ".join(processed_sentence.split())
    # 检测中文字符的正则表达式
    pattern_chinese = re.compile(r'[\\u4e00-\\u9fff]+')
    # 检测英文字符的正则表达式
    pattern_english = re.compile(r'[a-zA-Z]+')
    if pattern_english.findall(processed_sentence):
        processed_sentence = capitalize_first_letter_of_sentences(processed_sentence)
    if pattern_chinese.findall(processed_sentence):
        processed_sentence = processed_sentence.replace(" ", "")
    return processed_sentence
```

`capitalize_first_letter_of_sentences` 函数用于将输入文本中的每个句子的首字母大写。它首先通过 `split('. ')` 以 `.` 为分隔符将文本拆分成句子列表，然后去除空句子。接着，通过列表推导式将每个句子的首字母大写，并重新组合成一个新的文本。

`tokens_to_sequences` 函数用于处理预测得到的标记序列 `predicted_tokens` 。首先移除特定的标记 `[CLS]` 、 `[SEP]` 、 `[UNK]` 、 `[PAD]` 、 `[MASK]` ，然后去除多余的空格。之后，通过正则表达式检测文本中是否包含中文字符或英文字符，如果包含英文字符则调用 `capitalize_first_letter_of_sentences` 函数对文本进行首字母大写处理，如果包含中文字符则去除空格。最后返回处理后的文本。

```
def output_to_seq(output=None, tgt_indices=None, test=False):
    """将输出的概率矩阵解码成序列
    步骤：
        1、选择每个位置的最高概率词
        2、使用tokenizer解码
        3、处理解码结果得到字符串序列
    """
    if tgt_indices!=None:
        predicted_indices = tgt_indices
    else:
        if output==None:
            if test==True:
                output = torch.randn(2, 64, vocab_size)
            else:
                print("Error: The output is None. Please provide a valid output tensor.")
                return 
        print("OutputProbabilities:\n", output.size())
        # 选择每个位置的最高概率词
        predicted_indices = torch.argmax(output, dim=-1)
    print("OutputPredicatedIndices:\n", predicted_indices.size())
    print("OutputPredicatedSequences:")
    # 将索引转换为词，并转为字符串
    predicted_sequences = []
    for row in predicted_indices:
        predicted_tokens = tokenizer.decode(row)
        predicted_sequence = tokens_to_sequences(predicted_tokens)
        print(predicted_sequence)
        predicted_sequences.append(predicted_sequence)
    return predicted_sequences
```

这段代码定义了一个名为`output_to_seq`的函数，用于将模型输出的概率矩阵解码成序列。

1.  **输入参数处理**：
    
    *   `output`：模型输出的概率矩阵。
    *   `tgt_indices`：目标序列的索引。如果未提供，函数将尝试从`output`中提取。
    *   `test`：一个布尔值，指示是否处于测试模式。在测试模式下，如果`output`为`None`，函数将生成一个随机输出。
2.  **选择最高概率词**：
    
    *   如果`tgt_indices`未提供，函数将根据`output`来获取预测的索引。如果`output`为`None`且处于测试模式，函数将生成一个随机的输出矩阵。
    *   使用`torch.argmax(output, dim=-1)`来获取每个位置的最高概率词的索引。
3.  **解码和序列转换**：
    
    *   将获取到的索引使用`tokenizer.decode()`转换为对应的词。
    *   使用`tokens_to_sequences`函数进一步处理这些词，得到最终的字符串序列。
4.  **输出和返回**：
    
    *   打印解码后的序列，并将这些序列收集到一个列表中。
    *   最后，函数返回包含所有预测序列的列表。

#### 2.2.5 测试

```
def test():
    file_path = r"data\small\train.txt"
    loader, dataset = data_loader(file_path)
    """tokenizer的编码解码测试"""           
    print("编码:\n", dataset[0][0])
    decoded_text = tokenizer.decode(dataset[0][0])
    print("英文源文本:\n", dataset.data[0]['english'])
    print("解码:\n", decoded_text)
    print("序列化:\n", tokens_to_sequences(decoded_text))
    
    print("编码:\n", dataset[0][1])
    decoded_text = tokenizer.decode(dataset[0][1])
    print("中文源文本:\n", dataset.data[0]['chinese'])
    print("解码:\n",decoded_text)
    print("序列化:\n", tokens_to_sequences(decoded_text))
 
    """output_to_seq测试"""
    output_to_seq(test=True)
```

这段代码定义了一个名为`test`的函数，用于测试数据加载、tokenizer 的编码解码功能以及`output_to_seq`函数。以下是代码的详细解释：

1.  **数据加载**：
    
    *   `file_path`变量定义了训练数据的文件路径。
    *   `data_loader(file_path)`返回一个`loader`对象和`dataset`对象
    *   `dataset[0][0]`和`dataset[0][1]`假设是从`dataset`中获取的两个样本。
2.  **tokenizer 的编码解码测试**：
    
    *   打印第一对样本英文的编码结果。
    *   使用 tokenizer 的`decode`方法将编码结果解码回文本，并打印英文源文本。
    *   打印解码后的文本，并使用`tokens_to_sequences`函数将其序列化，然后打印序列化结果。
    *   重复上述步骤，但针对第一个样本对的中文进行输出。
3.  **output_to_seq 测试**：
    
    *   调用`output_to_seq(test=True)`函数进行测试。这里`test=True`意味着如果`output`为`None`，函数将生成一个随机的输出矩阵。

**测试结果**

![](https://i-blog.csdnimg.cn/direct/a051cc2b3a8346caa0c54121f85c2a8c.png)

![](https://i-blog.csdnimg.cn/direct/af28d56186144c3486a1e974207c1036.png)

我们可以看到，中英文序列会被编码为 tensor，每一个 token 对应一个索引，并且补齐至最大长度，解码后的输出需要经过处理才能得到正常的字符串序列。

![](https://i-blog.csdnimg.cn/direct/7b04e4242ad8490d9b2064b1cc73fa21.png)

 在测试下，会随机生成概率矩阵，接着通过最大概率获得索引矩阵，最后根据索引解码并处理得到字符串序列。由于是随机生成的内容，所以最终得到的序列为乱码。

好了，到目前为止，数据处理的工作差不多就完成了，接下来开始建模工作

三、模型
----

首先查看一下 GPU 信息，这点非常重要！

```
def get_gpu_info():
    # 查看GPU情况
    try:
        # 使用subprocess调用nvidia-smi命令
        result = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                                capture_output=True, text=True)
        output = result.stdout.strip().split('\\n')
        # 提取GPU信息
        gpu_info = {}
        for line in output:
            if line:
                name, memory = line.split(', ')
                gpu_info[name] = memory
                print(f"GPU型号: {name}, 内存: {memory}")
        print(f"GPU数量: {len(gpu_info)}")
    except Exception as e:
        print('Error:', str(e))
```

![](https://i-blog.csdnimg.cn/direct/66902e8d61354c88ba7f727a02c8b784.png)

这个结果意味着，不能进行多 GPU 分布式计算，不能用 Flash Attention 进行注意力计算加速（至少 RTX 3060），并且也不能用较大的 batch_size。过于贫穷......

> UserWarning: 1Torch was not compiled with flash attention. (Triggered internally at C:\cb\pytorch_1000000000000\work\aten\src\AT00\work\aten\src\ATen\native\transformers\cuda\sdp_utils.cpp:555.)

 这个是训练过程中的警告，但是硬件都不支持 Flash Attention，就没必要去配环境了（然而我搞了好久）

至于怎么用 GPU，需要 NVIDIA 显卡、cuda、cudNN、对应的 pytorch 版本等等，不是本篇博客的重点，请自行上网搜索，配置环境。

### 3.1 模型框架

![](https://i-blog.csdnimg.cn/direct/e44cc1b6532b45dea857bb110b09c1fa.png)

还是这张熟悉的图片，读完一部分核心源码之后，我本打算手搓一个编码器 - 解码器结构，但是考虑到时间成本和收益，我还是决定使用现有的编码器层、解码器层。

```
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
 
    def generate(self, src, beam_size=5, early_stopping=True)
```

**类定义：**这个类继承自`nn.Module`，用于创建一个 Transformer 模型。

```
class Transformer(nn.Module):

```

**构造函数`__init__：`**构造函数初始化模型的参数。

```
def __init__(self, vocab_size, embedding_dim, n_heads, n_layers, max_seq_length, dropout=0.1):

```

参数：

*   `vocab_size`：词汇表大小。
*   `embedding_dim`：嵌入维度。
*   `n_heads`：多头注意力的头数。
*   `n_layers`：编码器和解码器的层数。
*   `max_seq_length`：序列的最大长度。
*   `dropout`：丢弃率，用于防止过拟合，默认为 0.1。

**设备选择：**根据是否有可用的 GPU，选择设备（CPU 或 GPU）。

```
self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

```

**嵌入层：**创建一个嵌入层，用于将输入序列转换为嵌入向量。

```
self.embedding = nn.Embedding(vocab_size, embedding_dim).to(self.device)

```

**位置编码：**创建位置编码矩阵，用于在序列的每个位置上添加额外的信息。

```
self.positional_encoding = self.create_positional_encoding(embedding_dim, max_seq_length).to(self.device)

```

**编码器和解码器层：**使用了 PyTorch 的`TransformerEncoderLayer`和`TransformerDecoderLayer`。

```
encoder_layer = nn.TransformerEncoderLayer(embedding_dim, n_heads, dim_feedforward=2048, dropout=dropout, batch_first=True).to(self.device)
self.encoder = nn.TransformerEncoder(encoder_layer, n_layers).to(self.device)
decoder_layer = nn.TransformerDecoderLayer(embedding_dim, n_heads, dim_feedforward=2048, dropout=dropout, batch_first=True).to(self.device)
self.decoder = nn.TransformerDecoder(decoder_layer, n_layers).to(self.device)
```

**输出层：**创建一个线性层，用于将解码器输出转换为预测的输出分布。

```
self.fc_out = nn.Linear(embedding_dim, vocab_size).to(self.device)

```

**丢弃层：**创建一个丢弃层，用于在嵌入层和解码器输入层应用丢弃策略。

```
self.dropout = nn.Dropout(dropout).to(self.device)

```

**前向传播函数`forward`**

```
def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):

```

前向传播函数接收源序列`src`、目标序列`tgt`以及各种掩码（`src_mask`、`tgt_mask`、`memory_mask`），并执行以下操作：

1.  对源序列和目标序列进行嵌入。
2.  添加位置编码。
3.  对嵌入后的序列应用丢弃层。
4.  使用编码器对源序列进行编码。
5.  使用解码器对目标序列进行解码。
6.  通过输出层计算最终的输出。

**生成函数`generate`**

```
def generate(self, src, beam_size=5, early_stopping=True)

```

生成函数用于生成序列输出。接收源序列`src`和可选参数`beam_size`（用于贝叶斯搜索的大小）以及`early_stopping`（是否在生成过程中提前停止）。这个函数目前还没有调试成功。

### 3.2 其他工具

```
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
```

四、训练
----

```
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
```

这段代码定义了一个函数`model_train`，用于训练一个 Transformer 模型。以下是该函数的详细解释：

**参数**

*   `train_loader`: 训练数据的加载器。
*   `val_loader`: 验证数据的加载器。
*   `type`: 训练类型，0 代表英文到中文，1 代表中文到英文。
*   `num_epochs`: 训练的轮数，默认为 3。
*   `test`: 是否为测试模式，默认为 False。

**变量**

*   `total_training_time`: 总训练时间。
*   `loss_values`: 存储每个批次的损失值。
*   `validation_frequency`: 模型验证的频率。
*   `patience`: 早停机制的耐心值，默认为 5。
*   `best_val_loss`: 最佳验证损失，初始化为无穷大。
*   `patience_counter`: 早停计数器。

**训练流程**

1.  初始化模型、损失函数、优化器和调度器。
2.  在每个 epoch 中，遍历训练数据，进行前向传播、损失计算和反向传播。
3.  每隔一定批次，输出训练时间和平均损失。
4.  每隔`validation_frequency`批次，进行模型验证，并根据验证损失更新学习率和早停计数器。
5.  如果连续`patience`个 epoch 验证损失没有下降，则触发早停机制。

使用 small 数据训练这个模型（训练数据只有 960 条，batch_size=32，只有 30 个批次），设置 test=True 即为测试模型，epoch=10，每训练 10 个批次打印一次损失和时间，每 30 个批次进行一次验证，验证时不会对模型参数进行更新。

```
def train(train_data_path, val_data_path, test=False):
    print("----------模型训练测试----------")
    train_loader, train_dataset = data_loader(train_data_path)
    val_loader, val_dataset = data_loader(val_data_path)
    model_train(train_loader,val_loader,test=test)
 
if __name__=="__main__":
    train_data_path = r"data\small\train.txt"
    val_data_path = r"data\small\val.txt"
    train(train_data_path,val_data_path,test=True)
```

![](https://i-blog.csdnimg.cn/direct/29a3551f1e6947ac91dce1f6096da3e2.png)

![](https://i-blog.csdnimg.cn/direct/2661c4c94c684039b4690887e7e57a80.png)

![](https://i-blog.csdnimg.cn/direct/aa242cc6a0b049d793c405707b21c276.png)

![](https://i-blog.csdnimg.cn/direct/225c84d8cb464342b2a496363864c771.png)

从结果上看，模型成功跑起来了，loss 也有明显下降趋势。

五、评估
----

BLEU（Bilingual Evaluation Understudy）是一种用于评估机器翻译质量的指标。它是由 NIST（美国国家标准与技术研究院）提出的，用于衡量机器翻译的忠实度和流畅度。BLEU 得分通常介于 0 到 1 之间，得分越高表示翻译质量越好。

```
import torch
import model
import data_tool
import nltk
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
 
# 确保已经下载了nltk的corpus
# nltk.download('punkt')
 
def calculate_bleu_scores(translated_sentences, reference_sentences,type=0,test=False):
    bleu_scores = []     # 用于存储每个句子对的 BLEU 分数
    # 中英互译平滑函数选择
    if type==0:
        smoothing_function=SmoothingFunction().method4
    else:
        smoothing_function=SmoothingFunction().method2
    for i, (trans, ref) in enumerate(zip(translated_sentences, reference_sentences)):
        print(trans)
        print(ref)
        # 对翻译和参考句子进行分词
        trans_tokens = nltk.word_tokenize(trans)
        ref_tokens = nltk.word_tokenize(ref)
        # 计算句子对的 BLEU 分数，这里使用了默认的权重 (0.25, 0.25, 0.25, 0.25)
        bleu_score = sentence_bleu([ref_tokens], trans_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing_function)
        bleu_scores.append(bleu_score)
        if test==True: print(f"句子 {i+1} BLEU 分数: {bleu_score:.10f}")
    return bleu_scores
 
def bleu_test():
    references = ["我是学生", "我喜欢吃苹果"]
    hypothesis = ["我是学生", "我不喜欢吃苹果"]
    print("英译中：")
    scores = calculate_bleu_scores(hypothesis, references,0,True)
    # 计算平均BLEU分数
    average_score = sum(scores) / len(scores)
    print(f"所有生成翻译的平均BLEU分数为: {average_score}")
 
    references = ["This is a test sentence.", "I like apples."]
    hypothesis = ["This is a test sentence.", "I love apples."]
    print("中译英：")
    scores = calculate_bleu_scores(hypothesis, references,1,True)
    # 计算平均BLEU分数
    average_score = sum(scores) / len(scores)
    print(f"所有生成翻译的平均BLEU分数为: {average_score}")
```

 中英互译测试结果：

![](https://i-blog.csdnimg.cn/direct/f51c93326ecc48329efb07ec3d7bd9b0.png)

接下来我们使用测试数据集

```
def src_tgt_bleu_score(test_data_path, type=0, test=False):
    """在训练结束后,加载模型, 导入验证集, 将src和tgt输入模型,计算输出BLEU"""
    test_loader, test_dataset = model.data_loader(test_data_path)
    transformer = model.load_model(type,test)
    num_batchs = 0
    total_average_score = 0
    transformer.eval()
    with torch.no_grad():
        for English, Chinese in test_loader:
            num_batchs+=1
            test_src, test_tgt = model.get_src_tgt(English, Chinese, transformer, type)
            print(f"Batch {num_batchs} has {len(test_src)} samples")
            tgt_mask = model.create_tgt_mask(test_tgt, transformer)
            test_output = transformer(test_src, test_tgt, tgt_mask)
            references = data_tool.output_to_seq(tgt_indices=test_tgt)
            hypothesis = data_tool.output_to_seq(test_output)
            scores = calculate_bleu_scores(hypothesis, references,type)
            average_score = sum(scores) / len(scores)
            total_average_score += average_score
            print(f"平均BLEU分数为: {average_score}")
    print(f"所有生成翻译的平均BLEU分数为: {(total_average_score/num_batchs):.4f}")
```

训练 10 轮在测试集第 10 批次的结果：

![](https://i-blog.csdnimg.cn/direct/3db8d97db46249e793e61e04647f7ac6.png)

![](https://i-blog.csdnimg.cn/direct/9689dc606ad144f2becee0f8d8d22d81.png)

训练 5 轮测试结果：

![](https://i-blog.csdnimg.cn/direct/181ddeca29224f9281cc89f00bbd5ff4.png)
--------------------------------------------------------------------------

对比可得，在 small\train.txt 这个小数据集上，10 轮训练的结果远好于 5 轮训练的结果。并且，仔细观察不难看出，几乎只有完全预测正确的句子会得 1 分，存在个别错别字则为 0 分，导致平均得分非常低，我认为是我的 SmoothingFunction().method 选择问题。

六、总结
----

到目前为止，整个项目是一个半成品状态：

1.  generate 函数没有调试通过，希望此函数可以仅根据 src 序列生成 tgt 序列，并且完成测试
2.  模型没有在大规模的数据集进行训练
3.  模型的超参数还可以进一步调整
4.  src_mask 和 memory_mask 没有实现（影响不是特别大）

但是我很满意已经得到的成果，通过实际操作，我理解了 Transformer 模型的完整数据流，以及 tensor2tensor 模型的过程。代码我放在了 GitHub 仓库，有兴趣的朋友可以查看，也欢迎大家一起交流经验。