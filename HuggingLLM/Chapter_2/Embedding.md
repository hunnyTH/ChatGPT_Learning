# ChatGPT使用指南——相似匹配
## 1. Embedding
### 使用模型步骤
#### Step1： token化
* 按字
* 按词
* 按Bi-Gram
#### Step2：表示token
* one-Hot编码  

  问题：
    1. 数据维度太高：太高的维度会导致向量在空间中聚集在一个非常狭窄的角落，模型难以训练。
    2. 数据稀疏，向量之间缺乏语义上的交互（语义鸿沟）

* Embedding  
主要思想：
    1. 把特征固定在某一个维度D，避免维度过高的问题。
    2. 利用自然语言文本的上下文关系学习一个稠密表示。也就是说，每个Token的表示不再是预先算好的了，而是在过程中学习到的，元素也不再是很多个0，而是每个位置都有一个小数，这D个小数构成了一个Token表示。

#### 总结一下，Embedding本质就是一组稠密向量，用来表示一段文本（可以是字、词、句、段等），获取到这个表示后，我们就可以进一步做一些任务。

## 2. 相关API
Open AI  
智谱 AI
### 2.1 LMAS Embedding API
在自然语言处理领域，我们一般使用**cosine相似度**作为语义相似度的度量，评估两个向量在语义空间上的分布情况。
更多模型可以在这里查看：https://openai.com/blog/new-and-improved-embedding-model
### 2.2 ChatGPT Style

## 3. Embedding应用
### 3.1 QA
Q表示Question，A表示Answer  
关键点：
1. 事先需要有一个QA库。
2. 用户提问时，系统要能够在QA库中找到一个最相似的。

使用Kaggle提供的Quora数据集：https://www.kaggle.com/general/183270
### 3.2 聚类
### 3.3 推荐