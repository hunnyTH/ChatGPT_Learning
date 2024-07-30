import re
import json
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer
vocab_size = 30522  # 假设词汇表大小
max_seq_length = 128  # 最大序列长度
batch_size = 32 # 批处理长度

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

tokenizer = BertTokenizer.from_pretrained('bert-base-chinese')

def data_loader(data_path):
    data = read_file(data_path)
    # 创建测试数据集
    dataset = TranslationDataset(data, tokenizer)
    # 创建数据加载器
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return loader, dataset

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
    else:
        processed_sentence = processed_sentence.replace(" ", "").replace("。", "")
    return processed_sentence

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
        # 选择每个位置的最高概率词
        predicted_indices = torch.argmax(output, dim=-1)
    # 将索引转换为词，并转为字符串
    predicted_sequences = []
    for row in predicted_indices:
        predicted_tokens = tokenizer.decode(row)
        predicted_sequence = tokens_to_sequences(predicted_tokens)
        predicted_sequences.append(predicted_sequence)
    return predicted_sequences

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

if __name__=="__main__":
    test()