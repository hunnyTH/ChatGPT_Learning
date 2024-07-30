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
        smoothing_function=SmoothingFunction().method7
    else:
        smoothing_function=SmoothingFunction().method2
    for i, (trans, ref) in enumerate(zip(translated_sentences, reference_sentences)):
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


def generate_bleu_score(test_data_path, type=0, test=False):
    test_loader, test_dataset = model.data_loader(test_data_path)
    if type==0:
        src = test_dataset.English
        references = test_dataset.Chinese
    else:
        src = test_dataset.Chinese
        references = test_dataset.English
    hypothesis = []
    for row in src:
        output = model.predict(row,tokenizer=data_tool.tokenizer,test=True)
        hypothesis.append(output)
    # 计算BLEU分数
    scores = calculate_bleu_scores(hypothesis, references,type)
    average_score = sum(scores) / len(scores)
    print(f"所有生成翻译的平均BLEU分数为: {average_score:.4f}")

if __name__=="__main__":
    bleu_test()
    src_tgt_bleu_score(r"data\small\test.txt",test=True)
