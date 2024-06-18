Self-Attention机制，自注意力

Transformer是一种Encoder-Decoder架构，简单来说就是先把输入映射到Encoder
Transformer其实是Seq2Seq架构，序列到序列模型

Attention（注意力机制）：关注Encoder中Token的信息的机制，在生成每一个Token时都用到了Encoder每一个Token的信息，以及它已经生成的Token的信息。
GNMT的Attention，它是Decoder中的Token和Encoder中每一个Token的重要性权重。
Multi-Head Attention中用到SelfAttention，是自己的每一个Token和自己的每一个Token的重要性权重。

大多数NLP任务其实并不是Seq2Seq的，最常见的主要包括这么几种：句子级别分类、Token级别分类（也叫序列标注）、相似度匹配和生成

NLU（Natural Language Understanding，自然语言理解）任务
    句子级别分类是给定一个句子，输出一个类别
    Token级别的分类是给定一个句子，要给其中每个Token输出一个类别。
    NLU领域的第一个工作是Google的BERT，用了Transformer的Encoder架构，有12个Block，1亿多参数，它不预测下一个Token，而是随机把15%的Token盖住，然后利用其他没盖住的Token来预测盖住的Token。其实和根据上文预测下一个Token是类似的，不同的是可以利用下文信息。
    
NLG（Natural Language Generation，自然语言生成）任务
    生成、文本摘要、机器翻译、改写纠错等
    NLG领域的第一个工作是OpenAI的GPT，用的是Transformer的Decoder架构，参数和BERT差不多。

Transformer这个架构基于Seq2Seq，可以同时处理NLU和NLG任务，而且这种Self Attention机制的特征提取能力很强。这就使得NLP取得了阶段性的突破，深度学习开始进入了微调模型时代。大概的做法就是拿着一个开源的预训练模型，然后在自己的数据上微调一下，让它能够搞定特定的任务。


