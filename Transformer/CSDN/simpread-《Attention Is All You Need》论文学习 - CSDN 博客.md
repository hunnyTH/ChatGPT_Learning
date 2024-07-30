> 本文由 [简悦 SimpRead](http://ksria.com/simpread/) 转码， 原文地址 [blog.csdn.net](https://blog.csdn.net/T940842933/article/details/140440241?spm=1001.2014.3001.5502)

**目录**

[说明](#main-toc)

[摘要（Abstract）](#t0)

[1 介绍（Introduction）](#t1)

[2 背景（Background）](#t2)

[3 模型架构（Model Architecture）](#t3)

[3.1 编码器和解码器堆叠（Encoder and Decoder Stacks）](#t4)

[3.2 注意力机制（Attention）](#t5)

[3.2.1 缩放点积注意力（Scaled Dot-Product Attention）](#t6)

[3.2.2 多头注意力（Multi-Head Attention）](#t7)

[3.2.3 注意力机制在模型中的应用（Applications of Attention in our Model）](#t8)

[3.3 位置前馈网络（Position-wise Feed-forward Networks）](#t9)

[3.4 嵌入和 Softmax（Embedding and Softmax）](#t10)

[3.5 位置编码（Position Encoding）](#t11)

[4 为什么自注意（Why Self-Attention）](#t12)

[5 训练（Training）](#t13)

[5.1 训练数据和批处理（Training Data and Batching）](#t14)

[5.2 硬件和时间（Hardware and Schedule）](#t15)

[5.3 优化器（Optimizer）](#t16)

[5.4 规则（Regularization）](#t17)

[6 结果（Results）](#t18)

[6.1 机器翻译（Machine Translation）](#t19)

[6.2 模型变化（Model Variations）](#t20)

[7 结论（Conclusion）](#t21)

[学习总结](#t22)

说明
--

本篇博客记录学习论文《Attention Is All You Need》的全过程，不做任何商业用途，如有侵权请及时联系。

*   论文链接：[https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf](https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf "https://proceedings.neurips.cc/paper/2017/file/3f5ee243547dee91fbd053c1c4a845aa-Paper.pdf")
*   项目地址：[GitHub - tensorflow/tensor2tensor: Library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research.](https://github.com/tensorflow/tensor2tensor "GitHub - tensorflow/tensor2tensor: Library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research.")
*   学习工具：知云文献翻译 V8.4、Visual Studio Code、ChatGPT 3.5
*   学习内容：文献翻译、重点理解、写作技巧
*   学习目标：掌握 transformer 框架，理解模型算法过程

摘要（Abstract）
------------

> The dominant **sequence transduction models** are based on **complex recurrent** or **convolutional neural networks** that include an encoder and a decoder.

主流序列转导模型基于复杂的递归或卷积神经网络，包括编码器和[解码器](https://so.csdn.net/so/search?q=%E8%A7%A3%E7%A0%81%E5%99%A8&spm=1001.2101.3001.7020)。

**序列转导模型（sequence transduction models）：**

*   **输入输出序列映射**：序列转导模型的核心任务是学习将一个序列映射（或转换）成另一个序列。例如，将一段中文文本翻译成英文文本，或者将语音信号转换成文本序列。
    
*   **适用于多种任务**（包括但不限于）：
    
    *   机器翻译：将一种语言的序列翻译成另一种语言的序列。
    *   文本摘要：将长篇文本转换为短摘要的序列。
    *   语音识别：将语音信号转换为文本序列。
    *   命名实体识别：将文本序列中的命名实体标注出来。
    *   对话系统：将用户的自然语言输入转换成系统回应的序列。

> The best performing models also connect the encoder and decoder through an **attention mechanism**.

性能最好的模型还通过注意力机制连接编码器和解码器。

**注意力机制（attention mechanism）：**

*   **动态权重分配**：注意力机制允许模型在处理序列中的每个元素时，根据当前的上下文动态地给予不同元素不同的权重或注意力。这样模型可以集中精力处理对当前任务最重要的部分，而忽略不相关的部分，从而提高了模型的效率和性能。
*   **应用领域**：注意力机制最初是为了解决机器翻译中长距离依赖问题而提出的，但后来被广泛应用于各种序列到序列的转换任务，如文本摘要、语音识别、问答系统等。

> **We propose a new simple network architecture, the Transformer, based solely on attention mechanisms, dispensing with recurrence and convolutions entirely.**

我们提出了一种新的简单网络架构，Transformer，它完全基于注意力机制，完全省去了递归和卷积。

> Experiments on two machine translation tasks show these models to be **superior in quality** while being **more parallelizable** and requiring significantly **less time to train**.

在两个机器翻译任务上的实验表明，这些模型的质量更高，同时具有更高的并行性，并且需要更少的时间进行训练。

> Our model achieves 28.4 BLEU on the WMT 2014 Englishto-German translation task, improving over the existing best results, including ensembles, by over 2 BLEU.

我们的模型在 WMT 2014 英语到德语的翻译任务中达到了 28.4 BLEU，比现有的最佳结果（包括合奏）提高了 2 BLEU 以上。

> On the WMT 2014 English-to-French translation task, our model establishes a new single-model state-of-the-art BLEU score of 41.0 after training for 3.5 days on eight GPUs, a small fraction of the training costs of the best models from the literature.

在 WMT 2014 英语到法语翻译任务中，我们的模型在 8 个 GPU 上训练 3.5 天后建立了新的单模型最先进的 BLEU 分数 41.0，这是文献中最佳模型训练成本的一小部分。

**注：**BLEU（Bilingual Evaluation Understudy）是一种用于评估机器翻译质量的指标，它通过比较机器翻译的输出与人工翻译的参考翻译之间的相似度来评分。

**写作技巧**

摘要简洁明了，依次介绍了现有模型的传统框架、transformer 采用的的新型框架、transformer 的特点（优势），以及两个体现 transformer 性能的例子，使用了大量数据，直观感受。

1 介绍（Introduction）
------------------

**第一段：循环神经网络现状**

> Recurrent neural networks, **long short-term memory** [12] and **gated recurrent [7]** neural networks in particular, have been firmly established as state of the art approaches in sequence modeling and transduction problems such as language modeling and machine translation [29, 2, 5].

循环神经网络，特别是长短期记忆 [12] 和门控循环神经网络[7]，已被确立为序列建模和转导问题（如语言建模和机器翻译[29,2,5]）的最新方法。

> Numerous efforts have since continued to push the boundaries of recurrent language models and encoder-decoder architectures [31, 21, 13].

此后，许多努力继续扩展循环语言模型和[编码器 - 解码器](https://so.csdn.net/so/search?q=%E7%BC%96%E7%A0%81%E5%99%A8-%E8%A7%A3%E7%A0%81%E5%99%A8&spm=1001.2101.3001.7020)架构的边界 [31,21,13]。

**第二段：循神经环网络的机制、特点、问题**

> Recurrent models typically factor computation along the symbol positions of the input and output sequences.

循环模型通常根据输入和输出序列的符号位置进行因子计算。

> Aligning the positions to steps in computation time, they generate a sequence of hidden states ht, as a function of the previous hidden state ht−1 and the input for position t.

在计算时将位置与步骤对齐，它们会生成一系列隐藏状态 ht，作为先前隐藏状态 ht-1 和位置 t 输入的函数。

> This inherently sequential nature precludes **parallelization** within training examples, which becomes critical at longer sequence lengths, as memory constraints limit batching across examples.

这种固有的顺序性质阻止了训练示例中的并行化，这在较长的序列长度下变得至关重要，因为内存约束限制了跨示例的批处理。

**并行化（parallelization）**并行化是一种计算方法，它通过同时执行多个计算任务来提高程序的性能和效率。将一个大的任务分解成多个小任务，这些小任务可以同时在不同处理器或计算资源上执行。

在机器学习中，批处理（batching）是常见的优化手段，可以有效地利用 GPU 并行计算能力来加速训练过程。然而，对于序列数据，尤其是较长的序列，由于每个序列元素依赖于前面的元素，因此无法直接在多个序列之间实现并行化处理。这意味着，每个批次中的序列必须按顺序逐个处理，而不能同时处理多个序列。

> Recent work has achieved significant improvements in computational efficiency through **factorization tricks** [18] and **conditional computation** [26], while also improving model performance in case of the latter.

最近的工作通过因子分解技巧 [18] 和条件计算 [26] 显著提高了计算效率，同时也提高了后者的模型性能。

> The fundamental constraint of sequential computation, however, remains.

然而，顺序计算的基本约束仍然存在。

**第三段：注意力机制的特点**

> **Attention mechanisms** have become an integral part of compelling sequence modeling and transduction models in various tasks, **allowing modeling of dependencies without regard to their distance in the input or output sequences** [2, 16].

注意力机制已成为各种任务（引人注目的序列建模和转导模型）的重要组成部分，允许对依赖关系进行建模，而不考虑它们在输入或输出序列中的距离 [2,16]。

> In all but a few cases [22], however, such attention mechanisms are used in conjunction with a recurrent network.

然而，在除少数情况外的所有情况下 [22]，这种注意力机制都与循环网络结合使用。

**第四段：Transformer 的特点**

> In this work we propose the **Transformer, a model architecture eschewing recurrence and instead relying entirely on an attention mechanism to draw global dependencies between input and output.**

在这项工作中，我们提出了 Transformer，这是一种避免重复的模型架构，完全依赖于注意力机制来绘制输入和输出之间的全局依赖关系。

> The Transformer allows for significantly more parallelization and can reach a new state of the art in translation quality after being trained for as little as twelve hours on eight P100 GPUs.

Transformer 允许更多的并行化，并且在 8 个 P100 GPU 上训练了 12 个小时后，可以在翻译质量方面达到新的水平。

**写作技巧：**

首先提出循环神经网络是最新方法，在描述其机制时引出了并行性限制的问题，再通过介绍注意力机制不考虑输入输出序列距离的特点，从而体现 transformer 模型在并行性方面的优势，并且举例说明。

2 背景（Background）
----------------

**第一段：依赖关系学习**

> The goal of reducing sequential computation also forms the foundation of the Extended Neural GPU[20], ByteNet [15] and ConvS2S [8], all of which use convolutional neural networks as basic building block, computing hidden representations in parallel for all input and output positions.

减少顺序计算的目标也构成了扩展神经 GPU[20]、ByteNet[15] 和 ConvS2S[8] 的基础，所有这些都使用卷积神经网络作为基本构建块，并行计算所有输入和输出位置的隐藏表示。

> In these models, **the number of operations required to relate signals from two arbitrary input or output positions grows in the distance between positions**, linearly for ConvS2S and logarithmically for ByteNet.

在这些模型中，关联来自两个任意输入或输出位置的信号所需的操作数量随着位置之间的距离而增长，对于 ConvS2S 是线性增长的，对于 ByteNet 是对数增长的。

> This makes it more difficult to learn **dependencies** between distant positions [11].

这使得学习远距离位置之间的依赖关系变得更加困难 [11]。

**依赖关系（dependency）**指的是序列数据中各个元素之间的相互依赖或关联关系。这些依赖关系对于模型来说非常重要，因为它们决定了如何有效地理解和处理序列数据。

*   **时间依赖性**：序列数据通常具有时间上的依赖关系，即当前时刻的数据与之前时刻的数据相关联。例如，在语音识别中，理解当前词的语音片段可能需要考虑前面的声音上下文。
*   **空间依赖性**：除了时间上的依赖关系外，序列中的元素可能还存在空间上的依赖关系。例如，在自然语言处理中，一个单词的含义可能依赖于其前面几个单词的上下文。
*   **长距离依赖**：有些任务需要模型能够捕捉较长距离的依赖关系，即当前位置的预测可能依赖于序列中较远位置的信息。例如，机器翻译中，译文的某个词可能受源语言中较远位置的多个词影响。

> In the Transformer this is reduced to a constant number of operations, albeit at the cost of reduced effective resolution due to **averaging attention-weighted positions**, an effect we counteract with **Multi-Head Attention** as described in section 3.2.

在 Transformer 中，这被减少到恒定数量的操作，尽管由于平均注意力加权位置而降低了有效分辨率，但我们使用第 3.2 节中描述的多头注意力来抵消这种影响。

**第二段：自注意力**

> **Self-attention**, sometimes called intra-attention **is an attention mechanism relating different positions of a single sequence in order to compute a representation of the sequence.**

自我注意，有时也称为内部注意，是一种将单个序列的不同位置联系起来以计算序列表示的注意机制。

**自注意力（Self-attention）** 一种用于处理序列数据中长距离依赖关系的方法，能够有效地捕捉序列内部不同位置之间的关系，而无需依赖于序列的固定长度窗口或者固定的

*   **工作原理**：自注意力机制的核心在于，每个位置的输出不仅依赖于当前位置的输入，还依赖于所有其他位置的输入，且这种依赖关系是动态计算的。这使得模型能够在处理序列时，同时考虑到序列中不同位置的全局依赖关系，而不仅仅局限于固定的局部窗口。
*   **在 Transformer 中的应用**：Transformer 模型中的自注意力层允许模型同时处理输入序列的所有位置，使得模型能够更好地捕捉长距离依赖关系，从而在各种序列到序列的任务中取得了显著的性能提升。

> Self-attention has been used successfully in a variety of tasks including reading comprehension, abstractive summarization, textual entailment and learning task-independent sentence representations [4, 22, 23, 19].

自我注意已被成功应用于各种任务，包括阅读理解、抽象概括、文本蕴涵和独立于学习任务的句子表征 [4,22,23,19]。

**第三段：端到端记忆网络**

> **End-to-end memory networks** are based on a recurrent attention mechanism instead of sequencealigned recurrence and have been shown to perform well on simple-language question answering and language modeling tasks [28].

端到端记忆网络基于循环注意力机制，而不是序列设计的循环，并且已被证明在简单的语言问答和语言建模任务中表现良好 [28]。

**第四段：transformer 描述**

> To the best of our knowledge, however, **the Transformer is the first transduction model relying entirely on self-attention to compute representations of its input and output** without using sequencealigned RNNs or convolution.

然而，据我们所知，Transformer 是第一个完全依赖自我关注来计算其输入和输出表示的转导模型，而不使用序列设计的 RNN 或卷积。

> In the following sections, we will describe the Transformer, motivate self-attention and discuss its advantages over models such as [14, 15] and [8].

在接下来的部分中，我们将描述 Transformer，激发自我关注，并讨论它相对于 [14,15] 和[8]等模型的优势。

**写作技巧**

1.  第一段：现有模型依赖关系学习困难，操作数会随位置距离增大而增多；transformer 固定操作数量，使用平均注意力加权位置和多头注意力机制
2.  第二、三段：自注意力机制及其应用情况；端到端记忆网络机制与应用情况
3.  第四段：Transformer 是第一个完全依赖自我关注来计算其输入和输出表示的转导模型；引出下文

写作背景包括模型现存问题、解决方法、已有方法描述、创新方法描述等，并为下文做铺垫

3 模型架构（Model Architecture）
--------------------------

**第一段：编码器 - 解码器结构原理**

> Most competitive neural sequence transduction models have an **encoder-decoder structure** [5, 2, 29].

大多数竞争性神经序列转导模型具有编码器 - 解码器结构 [5,2,29]。

> Here, the encoder maps an input sequence of **symbol representations** $(x_{1}, ... , x_{n})$ to a sequence of **continuous representations**  $z = (z_{1}, ... , z_{n})$.

这里，编码器将符号表示的输入序列 $(x_{1}, ... , x_{n})$ 映射到连续表示的序列 $z = (z_{1}, ... , z_{n})$。

> Given z, the decoder then generates an output sequence $(y_{1}, ... , y_{m})$ of **symbols** one element at a time.

给定 $z$，解码器然后一次生成一个元素的符号输出序列 $(y_{1}, ... , y_{m})$。

> At each step the model is **auto-regressive**[9], consuming the previously generated symbols as additional input when generating the next.

在每一步，模型都是自回归的 [9]，在生成下一步时，将之前生成的符号作为额外的输入。

**自回归（auto-regressive）**一种统计和时间序列分析中常用的模型形式，它假设当前值与过去的值存在某种线性关系。

**第二段：Transformer 模型结构**

> The Transformer follows this overall architecture using **stacked self-attention** **and** **point-wise**, **fully connected layers** for both the encoder and decoder, shown in the left and right halves of Figure 1, respectively.

Transformer 遵循这种整体架构，为编码器和解码器使用堆叠的自注意力和逐点、完全连接层，分别如图 1 的左半部分和右半部分所示。

![](https://i-blog.csdnimg.cn/direct/aa150e412bd348a6bc5ba5a000e4e49b.png)​​

Figure 1: The Transformer - model architecture.

**部分模块简单说明**

假设我们有一个时间序列数据集，它包含一系列观测值：`X = [x1, x2, x3, ..., xN]`。

**Inputs：**输入序列 [x1, x2, x3, x4, x5]

**Outputs（Shifted right）：**输出序列（输入序列向右移位）[x0, x1, x2, x3, x4]

**Embedding：**将高维数据映射到低维空间

**Positional Encoding：**由于 Transformer 不像 RNNs 那样固有地捕捉序列中元素的顺序，因此添加位置编码以提供关于输入序列中单词或标记位置的信息。

**Add & Norm：**

*   加法连接（Addition）：在每个子层的输出之后，将该子层的输入（或原始输入）与子层的输出进行加法连接。这种加法连接称为残差连接（residual connection），
*   层归一化（Layer Normalization）：  在进行加法连接后，对得到的向量进行层归一化操作。层归一化是一种归一化技术，它独立地对每个样本的特征进行归一化，以确保每个特征的均值为 0、方差为 1。

**Feed forward (Neural Network)：**一种人工神经网络模型，其中信息流动的方向是单向的，即从输入层流向输出层，没有循环或反馈连接

**Linear：**线性变换

**SoftMax：**将一个 K 维的实数向量（通常称为 logits 或 scores）转换为一个概率分布，其中每个元素的取值范围在 0 到 1 之间，并且所有元素的和为 1。

**Output Probabilities：**一个概率分布，表示每个可能的输出标记（如词语或标签）的概率。

### 3.1 编码器和解码器堆叠（Encoder and Decoder Stacks）

> **Encoder:** The encoder is composed of a stack of **N = 6** identical layers. Each layer has two sub-layers. The first is a **multi-head self-attention mechanism**, and the second is a simple, position-wise fully connected **feed-forward network**. We employ a **residual connection** [10] around each of the two sub-layers, followed by **layer normalization** [1]. That is, the output of each sub-layer is LayerNorm(x + Sublayer(x)), where Sublayer(x) is the function implemented by the sub-layer itself. To facilitate these residual connections, all sub-layers in the model, as well as the embedding layers, produce outputs of dimension **dmodel = 512**.

**编码器：**编码器由 $N=6$ 个相同层的堆栈组成。每一层有两个子层。第一层是[多头自注意力机制](https://so.csdn.net/so/search?q=%E5%A4%9A%E5%A4%B4%E8%87%AA%E6%B3%A8%E6%84%8F%E5%8A%9B%E6%9C%BA%E5%88%B6&spm=1001.2101.3001.7020)，第二层是简单的位置 - 智能全连接前馈网络。我们在两个子层的每一个周围使用残差连接 [10]，然后进行层归一化 [1]。也就是说，每个子层的输出是 LayerNorm(x + 子层（x）)，其中子层 (x) 是子层本身实现的函数。为了促进这些残余连接，模型中的所有子层以及嵌入层都会产生维度 dmodel=512 的输出。

> **Decoder:** The decoder is also composed of a stack of **N = 6** identical layers. In addition to the two sub-layers in each encoder layer, **the decoder inserts a third sub-layer, which performs multi-head attention over the output of the encoder stack**. Similar to the encoder, we employ residual connections around each of the sub-layers, followed by layer normalization. We also modify the self-attention sub-layer in the decoder stack to prevent positions from attending to subsequent positions. This **masking**, combined with fact that the output embeddings are offset by one position, ensures that the predictions for position i can depend only on the known outputs at positions less than i.

**解码器：**解码器也由 $N=6$ 个相同层的堆栈组成。除了每个编码器层中的两个子层外，解码器还插入了第三个子层，该子层对编码器堆栈的输出执行多头注意力。与编码器类似，我们在每个子层周围使用残差连接，然后进行层归一化。我们还修改了解码器堆栈中的自注意子层，以防止位置关注后续位置。这种掩蔽，再加上输出嵌入偏移了一个位置的事实，确保了位置 i 的预测只能依赖于小于 i 的位置处的已知输出。

动态图（N =  2） ：

![](https://i-blog.csdnimg.cn/direct/26a31e795b9e4662bc54a7763b0b3a78.gif)​​

### 3.2 注意力机制（Attention）

> An attention function can be described as **mapping** a query and a set of key-value pairs to an output, where the **query, keys, values**, and output are all vectors.

注意力函数可以描述为将查询和一组键值对映射到输出，其中查询、键、值和输出都是向量。

> The output is computed as a weighted sum of the **values**, where the weight assigned to each value is computed by a compatibility function of the query with the corresponding key.

值的加权和为输出，其中分配给每个值的权重是由查询与相应键的兼容性函数计算的。

#### 3.2.1 缩放点积注意力（Scaled Dot-Product Attention）

**第一段：缩放点击注意力机制计算过程**

> We call our particular attention "**Scaled Dot-Product Attention**" (Figure 2). The input consists of queries and keys of dimension $d_{k}$, and values of dimension $d_{v}$.

我们特别关注 “缩放点积注意力”（图 2）。输入由维度 $d_{k}$的查询和键以及维度 $d_{v}$的值组成。

> We compute the dot products of the query with all keys, divide each by $\sqrt{d_{k}}$, and apply a softmax function to obtain the weights on the values.

我们计算包含所有键的查询的点积，将每个键除以 $\sqrt{d_{k}}$ ，并应用 softmax 函数来获得值的权重。

![](https://i-blog.csdnimg.cn/direct/56462d14ca224dd895e5fa9cf7f8e0f2.png)​

Figure 2: Scaled Dot-Product Attention. 

**模块说明及步骤：**

1.  **查询（Query）、键（Key）、值（Value）**：自注意力机制通过计算输入序列的查询、键和值来工作。这些查询、键和值都是通过对输入序列进行线性变换获得的。
2.  **MatMul：**计算 Query 矩阵 Q、Key 矩阵 K 的乘积，得到得分矩阵 scores。
3.  **Scale：**对得分矩阵 scores 进行缩放，即将其除以向量维度的平方根（np.sqrt(d_k)）
4.  **Mask(opt.)：**若存在 Attention Mask，则将 Attention Mask 的值为 True 的位置对应的得分矩阵元素置为负无穷（-inf）。
5.  **Softmax：**对得分矩阵 scores 进行 softmax 计算，得到 Attention 权重矩阵 attn。
6.  **MatMul：**计算 Attention 权重矩阵 attn 和 Value 矩阵 V 的乘积，得到加权后的 Context 矩阵。

**第二段：注意力分数计算**

>  In practice, we compute the attention function on a set of queries simultaneously, packed together into a matrix Q. The keys and values are also packed together into matrices K and V . We compute the matrix of outputs as:

在实际应用中，我们同时计算一组查询的注意力函数，并将其打包成矩阵 Q。键和值也打包成矩阵 K 和 V。我们计算输出矩阵如下：

$Attention(Q, K, V ) = softmax(\frac{QK^{T}}{\sqrt{d_{k}}})V$

**第三段：注意力函数——加或乘**

> The two most commonly used attention functions are additive attention [2], and dot-product (multiplicative) attention.

最常用的两种注意力函数是加法注意力 [2] 和点积（乘）注意力。

> Dot-product attention is identical to our algorithm, except for the scaling factor of $\frac{1}{\sqrt{d_{k}}}$.

我们的算法使用了点积注意力，除了缩放因子 $\frac{1}{\sqrt{d_{k}}}$。

> Additive attention computes the compatibility function using a feed-forward network with  
> a single hidden layer.

加法注意力使用具有单个隐藏层的前馈网络来计算兼容性函数。

> While the two are similar in theoretical complexity, dot-product attention is much faster and more space-efficient in practice, since it can be implemented using highly optimized matrix multiplication code.

虽然两者在理论复杂性上相似，但点积注意力在实践中要快得多，空间效率也更高，因为它可以使用高度优化的矩阵乘法代码来实现。

**第四段：$d_{k}$ 对注意力函数的影响**

> While for small values of $d_{k}$ the two mechanisms perform similarly, additive attention outperforms dot product attention without scaling for larger values of $d_{k}$ [3].

虽然对于较小的 $d_{k}$ 值，这两种机制的表现相似，但在不缩放较大 $d_{k}$ 值的情况下，加法注意力优于点积注意力 [3]。

> We suspect that for large values of $d_{k}$, the dot products grow large in magnitude, pushing the softmax function into regions where it has extremely small gradients. To counteract this effect, we scale the dot products by $\frac{1}{\sqrt{d_{k}}}$ .

我们怀疑，对于较大的 $d_{k}$ 值，点积的幅度会增大，将 softmax 函数推入梯度极小的区域。为了抵消这种影响，我们将点积缩放 $\frac{1}{\sqrt{d_{k}}}$。

> annotation: To illustrate why the dot products get large, assume that the components of $q$ and $k$ are independent random variables with mean 0 and variance 1. Then their dot product, $q \cdot k =\sum_{i=1}^{d_{k}}q_{i}k_{i}$, has mean 0 and variance $d_{k}$.

注：为了说明点积变大的原因，假设 $q$和 $k$的分量是均值为 0、方差为 1 的独立随机变量。那么它们的点积 $q \cdot k =\sum_{i=1}^{d_{k}}q_{i}k_{i}$ 的均值为 0，方差为 $d_{k}$。

#### 3.2.2 多头注意力（Multi-Head Attention）

**第一段：多头注意力机制计算过程**

> Instead of performing a single attention function with $d_{model}$-dimensional keys, values and queries, we found it beneficial to linearly project the queries, keys and values $h$ times with different, learned linear projections to $d_{k}$, $d_{k}$ and $d_{v}$ dimensions, respectively.

我们发现，比起用 $d_{model}$维度的键、值和查询执行单个注意力函数，将查询、键和值分别线性投影到 $d_{k}$、$d_{k}$ 和 $d_{v}$ 维度的不同学习线性投影 $h$ 次是有益的。

> On each of these projected versions of queries, keys and values we then perform the attention function in parallel, yielding $d_{v}$-dimensional output values.

然后，在查询、键和值的每个投影版本上，我们并行执行注意力函数，产生 $d_{v}$ 维输出值。

> These are concatenated and once again projected, resulting in the final values, as depicted in Figure 2.

这些值被连接并再次投影，得到最终值，如图 2 所示。

![](https://i-blog.csdnimg.cn/direct/e9529b997c8d4428b40cb89dbd95c1ca.png)​

Figure 2: Multi-Head Attention consists of several attention layers running in parallel.

**模块说明及步骤：**

1.  **查询（Query）、键（Key）、值（Value）：**Q 通过对输出序列进行线性变换或前一层 Decoder 获得，V、K 通过 6 层 Encoder 编码获得。
2.  **Linear：**将 Q、K、V 经过 $W_{i}^{Q}, W_{i}^{K}, W_{i}^{V}$ 映射到 $Q_{i}, K_{i}, V_{i}$，其中 $i=1, 2, ... , h$
3.  **Scaled Dot-Product Attention：**对每个 Head 分别进行缩放点积注意力计算
4.  **Concat：**拼接多头注意力分数
5.  **Linear：**将拼接后的结果通过 $W^{O}$（可学习的参数）进行映射

**第二段：多头注意力公式**

> Multi-head attention allows the model to jointly attend to information from different representation subspaces at different positions. With a single attention head, averaging inhibits this.

多头注意力允许模型在不同位置联合处理来自不同表示子空间的信息。如果只有一个注意力，平均值会抑制这种情况。

$MultiHead(Q, K, V ) = Concat(head_{1},...,head_{h})W^O$

$where \ head_{i} = Attention(QW_{i}^{Q}; KW_{i}^{K}; VW_{i}^{V})$

> Where the projections are parameter matrices $W_{i}^{Q}\in \mathbb{R}^{d_{model}\times d_{k}}$, $W_{i}^{K}\in \mathbb{R}^{d_{model}\times d_{k}}$, $W_{i}^{V}\in \mathbb{R}^{d_{model}\times d_{v}}$ and $W^{O}\in \mathbb{R}^{hd_{v}\times d_{model}}$.

其中投影是参数矩阵 $W_{i}^{Q}\in \mathbb{R}^{d_{model}\times d_{k}}$, $W_{i}^{K}\in \mathbb{R}^{d_{model}\times d_{k}}$, $W_{i}^{V}\in \mathbb{R}^{d_{model}\times d_{v}}$和 $W^{O}\in \mathbb{R}^{hd_{v}\times d_{model}}$

**第三段：模型超参数**

> In this work we employ $h = 8$ parallel attention layers, or heads. For each of these we use $d_{k} = d_{v} = d_{model}/h = 64$.

在这项工作中，我们使用了 $h = 8$个平行的注意力层或头部。对于每种情况，我们使用 $d_{k} = d_{v} = d_{model}/h = 64$。

> Due to the reduced dimension of each head, the total computational cost is similar to that of single-head attention with full dimensionality.

由于每个头部的尺寸减小，总计算成本与全维单头注意力相似。

#### 3.2.3 注意力机制在模型中的应用（Applications of Attention in our Model）

> The Transformer uses multi-head attention in three different ways:
> 
> *   In "encoder-decoder attention" layers, the queries come from the previous decoder layer, and the memory keys and values come from the output of the encoder. This allows every position in the decoder to attend over all positions in the input sequence. This mimics the typical encoder-decoder attention mechanisms in sequence-to-sequence models such as [31, 2, 8].
> *   The encoder contains self-attention layers. In a self-attention layer all of the keys, values and queries come from the same place, in this case, the output of the previous layer in the encoder. Each position in the encoder can attend to all positions in the previous layer of the encoder.
> *   Similarly, self-attention layers in the decoder allow each position in the decoder to attend toall positions in the decoder up to and including that position. We need to prevent leftward information flow in the decoder to preserve the auto-regressive property. We implement this inside of scaled dot-product attention by masking out (setting to $-\infty$) all values in the input of the softmax which correspond to illegal connections. See Figure 2.

Transformer 以三种不同的方式使用多头注意力：

*   在 “编码器 - 解码器注意” 层中，解码层的查询 Q 来自前一个解码器层，内存键 K 和值 V 来自编码器的输出。这允许解码器中的每个位置都能处理输入序列中的所有位置。这模仿了典型的编码器 - 解码器注意力机制，如[31,2,8]。
*   编码器包含自我关注层。在自我关注层中，所有键、值和查询都来自同一个地方，在这种情况下，是编码器中前一层的输出。编码器中的每个位置都可以处理编码器前一层中的所有位置
*   类似地，解码器中的自我关注层允许解码器中的每个位置关注解码器中的所有位置，直到并包括该位置。我们需要防止解码器中的向左信息流，以保持自回归特性。我们通过屏蔽（设置为$-\infty$）输入中与非法连接对应的所有值来实现缩放点积内注意力。见图 2。

### 3.3 位置前馈网络（Position-wise Feed-forward Networks）

> In addition to attention sub-layers, each of the layers in our encoder and decoder contains a fully connected feed-forward network, which is applied to each position separately and identically. This consists of two linear transformations with a ReLU activation in between.

除了注意子层外，我们编码器和解码器中的每一层都包含一个完全连接的前馈网络，该网络分别且相同地应用于每个位置。这由两个线性变换组成，中间有一个 ReLU 激活。

$FFN(x) = max(0, xW_{1} + b_{1})W_{2} + b_{2}$

> While the linear transformations are the same across different positions, they use different parameters from layer to layer. Another way of describing this is as two convolutions with kernel size 1.

虽然不同位置的线性变换是相同的，但它们在不同层之间使用不同的参数。另一种描述方式是将其描述为核大小为 1 的两个卷积。

> The dimensionality of input and output is $d_{model} = 512$, and the inner-layer has dimensionality $d_{ff} = 2048$.

输入和输出的维度为 $d_{model} = 512$，内层的维度为 $d_{ff} = 2048$。

**作用：**

1.  增加模型的表达能力：通过引入非线性变换，能够学习到输入特征之间更复杂的关系，从而丰富模型对数据的表示。
2.  对每个位置进行独立的特征变换：在处理序列数据时，能够针对序列中的每个位置进行特定的特征提取和转换，不受其他位置的影响。
3.  补充注意力机制的不足：注意力机制主要关注了不同位置之间的关系，而前馈神经网络可以对每个位置的特征进行更深入的加工和提炼。
4.  捕捉局部特征：有助于捕捉输入数据中的局部模式和细节信息，与注意力机制侧重于全局依赖形成互补。
5.  加速训练和提高稳定性：适当的前馈神经网络结构和参数设置可以帮助模型更快地收敛，并且在训练过程中提高稳定性，减少过拟合的风险。

### 3.4 嵌入和 Softmax（Embedding and Softmax）

> Similarly to other sequence transduction models, we use learned **embeddings** to convert the input tokens and output tokens to vectors of dimension $d_{model}$.

与其他序列转导模型类似，我们使用学习的嵌入将输入标记和输出标记转换为维度 $d_{model}$ 的向量。

**Embedding**：在自然语言处理中，文本通常以单词或短语的形式存在，这些数据具有高维度和稀疏性。通过词嵌入（Word Embedding）技术，每个单词或短语可以被映射到一个低维的实数空间中，使得具有相似含义的单词在向量空间中也彼此接近。

> We also use the usual learned linear transformation and softmax function to convert the decoder output to predicted next-token probabilities.

我们还使用通常的学习线性变换和 softmax 函数将解码器输出转换为预测的下一个令牌概率。

> In our model, we share the same weight matrix between the two embedding layers and the pre-softmax linear transformation, similar to [24]. In the embedding layers, we multiply those weights by $\sqrt{d_{model}}$.

在我们的模型中，我们在两个嵌入层和前 softmax 线性变换之间共享相同的权重矩阵，类似于 [24]。在嵌入层中，我们将这些权重乘以 $\sqrt{d_{model}}$。

### 3.5 位置编码（Position Encoding）

> Since our model contains no recurrence and no convolution, in order for the model to make use of the order of the sequence, we must inject some information about the relative or absolute position of the tokens in the sequence.

由于我们的模型不包含递归和卷积，为了使模型利用序列的顺序，我们必须注入一些关于序列中标记的相对或绝对位置的信息。（自注意力机制本身无法感知单词的顺序）

> To this end, we add "positional encodings" to the input embeddings at the bottoms of the encoder and decoder stacks.

为此，我们在编码器和解码器堆栈底部的输入嵌入中添加 “位置编码”

> The positional encodings have the same dimension $d_{model}$ as the embeddings, so that the two can be summed. There are many choices of positional encodings, learned and fixed [8].

位置编码与嵌入具有相同的维度 $d_{model}$，因此可以将两者相加。位置编码有很多选择，可学习编码和固定编码 [8]。

Table 1: Maximum path lengths, per-layer complexity and minimum number of sequential operations for different layer types. $n$ is the sequence length, $d$ is the representation dimension, $k$ is the kernel size of convolutions and $r$ the size of the neighborhood in restricted self-attention.

![](https://i-blog.csdnimg.cn/direct/19eae61f3ed54330b0fc2f23197f66c8.png)​表 1：不同层类型的最大路径长度、每层复杂性和最小顺序操作数。$n$ 是序列长度，$d$ 是表示维数，$k$ 是卷积的核大小，$r$ 是受限自注意中邻域的大小。

> In this work, we use sine and cosine functions of different frequencies:

在这项工作中，我们使用不同频率的正弦和余弦函数：

$P E_{pos,2i} = sin(pos/10000^{2i/d_{model}})$

$P E_{pos,2i+1} = cos(pos/10000^{2i/d_{model}})$

> where $pos$ is the position and $i$ is the dimension. That is, each dimension of the positional encoding corresponds to a sinusoid. The wavelengths form a geometric progression from $2\pi$ to $10000\cdot 2\pi$.

其中 $pos$ 是位置，$i$ 是维度。也就是说，位置编码的每个维度都对应于一个正弦曲线。波长形成从 $2\pi$ 到 $10000\cdot 2\pi$ 的几何级数。

We chose this function because we hypothesized it would allow the model to easily learn to attend by relative positions, since for any fixed offset $k$, $PE_{pos+k}$ can be represented as a linear function of $PE_{pos}$.

我们选择这个函数是因为我们假设它可以让模型很容易地学会通过相对位置来参与，因为对于任何固定的偏移量 $k$，$PE_{pos+k}$ 都可以表示为 $PE_{pos}$ 的线性函数。

4 为什么自注意（Why Self-Attention）
----------------------------

> In this section we compare various aspects of self-attention layers to the recurrent and convolutional layers commonly used for mapping one variable-length sequence of symbol representations $(x_{1}, ... , x_{n})$ to another sequence of equal length $(z_{1}, ... , z_{n})$, with $x_{i},z_{i} \in \mathbb{R}^{d}$, such as a hidden layer in a typical sequence transduction encoder or decoder. Motivating our use of self-attention we consider three desiderata.

在本节中，我们将自注意层的各个方面与通常用于将符号表示的一个可变长度序列 $(x_{1}, ... , x_{n})$ 映射到另一个相等长度序列 $(z_{1}, ... , z_{n})$ 的递归层和卷积层进行比较，其中 $x_{i},z_{i} \in \mathbb{R}^{d}$；例如典型序列转导编码器或解码器中的隐藏层。激励我们使用自我关注，我们认为有三个必要条件。

> One is the total computational complexity per layer. Another is the amount of computation that can be parallelized, as measured by the minimum number of sequential operations required.

一个是每层的总计算复杂度。另一个是可以并行化的计算量，通过所需的最小顺序操作数来衡量。

> The third is the path length between long-range dependencies in the network. Learning long-range dependencies is a key challenge in many sequence transduction tasks. One key factor affecting the ability to learn such dependencies is the length of the paths forward and backward signals have to traverse in the network. The shorter these paths between any combination of positions in the input and output sequences, the easier it is to learn long-range dependencies [11]. Hence we also compare the maximum path length between any two input and output positions in networks composed of the different layer types.

第三个是网络中长程依赖之间的路径长度。学习长程依赖是许多序列转导任务中的一个关键挑战。影响学习这种依赖关系能力的一个关键因素是网络中前向和后向信号必须经过的路径长度。输入和输出序列中任何位置组合之间的路径越短，就越容易学习长距离依赖关系 [11]。因此，我们还比较了由不同层类型组成的网络中任意两个输入和输出位置之间的最大路径长度。

> As noted in Table 1, a self-attention layer connects all positions with a constant number of sequentially executed operations, whereas a recurrent layer requires O(n) sequential operations. In terms of computational complexity, self-attention layers are faster than recurrent layers when the sequence length n is smaller than the representation dimensionality d, which is most often the case with sentence representations used by state-of-the-art models in machine translations, such as word-piece [31] and byte-pair [25] representations. To improve computational performance for tasks involving very long sequences, self-attention could be restricted to considering only a neighborhood of size r in 6the input sequence centered around the respective output position. This would increase the maximum path length to O(n=r). We plan to investigate this approach further in future work.

如表 1 所示，自我关注层通过恒定数量的顺序执行操作连接所有位置，而循环层需要 O（n）个顺序操作。就计算复杂性而言，当序列长度 n 小于表示维度 d 时，自关注层比循环层更快，这在机器翻译中最先进的模型使用的句子表示中最为常见，如单词段 [31] 和字节对 [25] 表示。为了提高涉及很长序列的任务的计算性能，可以将自我注意力限制在只考虑 6 中以相应输出位置为中心的输入序列中大小为 r 的邻域。这将使最大路径长度增加到 O（n=r）。我们计划在未来的工作中进一步研究这种方法。

> A single convolutional layer with kernel width k <n does not connect all pairs of input and output positions. Doing so requires a stack of O(n=k) convolutional layers in the case of contiguous kernels, or O(logk(n)) in the case of dilated convolutions [15], increasing the length of the longest paths between any two positions in the network. Convolutional layers are generally more expensive than recurrent layers, by a factor of k. Separable convolutions [6], however, decrease the complexity considerably, to O(k · n · d + n · d²). Even with k = n, however, the complexity of a separable convolution is equal to the combination of a self-attention layer and a point-wise feed-forward layer, the approach we take in our model.

具有核宽度 k<n 的单个卷积层不会连接所有输入和输出位置对。这样做在连续核的情况下需要一堆 O（n=k）卷积层，在扩展卷积的情况下则需要 O（logk（n））[15]，这增加了网络中任意两个位置之间最长路径的长度。卷积层通常比循环层贵 k 倍。然而，可分离卷积 [6] 大大降低了复杂性，达到 O（k·n·d+n·d²）。然而，即使 k=n，可分离卷积的复杂性也等于我们在模型中采用的自我关注层和逐点前馈层的组合。

> As side benefit, self-attention could yield more interpretable models. We inspect attention distributions from our models and present and discuss examples in the appendix. Not only do individual attention heads clearly learn to perform different tasks, many appear to exhibit behavior related to the syntactic and semantic structure of the sentences.

作为附带好处，自我关注可以产生更多可解释的模型。我们从模型中检查注意力分布，并在附录中给出和讨论示例。个体注意头不仅清楚地学会了执行不同的任务，而且许多人似乎表现出与句子的句法和语义结构相关的行为。

5 训练（Training）
--------------

### 5.1 训练数据和批处理（Training Data and Batching）

### 5.2 硬件和时间（Hardware and Schedule）

### 5.3 优化器（Optimizer）

### 5.4 规则（Regularization）

6 结果（Results）
-------------

### 6.1 机器翻译（Machine Translation）

### 6.2 模型变化（Model Variations）

7 结论（Conclusion）
----------------

> In this work, we presented the Transformer, the first sequence transduction model based entirely on attention, replacing the recurrent layers most commonly used in encoder-decoder architectures with multi-headed self-attention.

在这项工作中，我们提出了 Transformer，这是第一个完全基于注意力的序列转导模型，用多头自注意力取代了编码器 - 解码器架构中最常用的循环层

> For translation tasks, the Transformer can be trained significantly faster than architectures based on recurrent or convolutional layers. On both WMT 2014 English-to-German and WMT 2014 English-to-French translation tasks, we achieve a new state of the art. In the former task our best model outperforms even all previously reported ensembles.

对于翻译任务，Transformer 的训练速度明显快于基于循环层或卷积层的架构。在 WMT 2014 英语到德语和 WMT 2014 法语到英语的翻译任务中，我们实现了新的技术水平。在前一个任务中，即使是之前报道的所有组合，我们的最佳模型也表现出色。

> We are excited about the future of attention-based models and plan to apply them to other tasks. We plan to extend the Transformer to problems involving input and output modalities other than text and to investigate local, restricted attention mechanisms to efficiently handle large inputs and outputs such as images, audio and video. Making generation less sequential is another research goals of ours.

我们对基于注意力的模型的未来感到兴奋，并计划将其应用于其他任务。我们计划将 Transformer 扩展到涉及文本以外的输入和输出模式的问题，并研究局部、受限的注意力机制，以有效地处理图像、音频和视频等大型输入和输出。减少代际顺序是我们的另一个研究目标。

> The code we used to train and evaluate our models is available at https://github.com/tensorflow/tensor2tensor.

我们用来训练和评估模型的代码在 [GitHub - tensorflow/tensor2tensor: Library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research.](https://github.com/tensorflow/tensor2tensor "GitHub - tensorflow/tensor2tensor: Library of deep learning models and datasets designed to make deep learning more accessible and accelerate ML research.")

学习总结
----

 《Attention Is All You Need》是 2017 年由 Vaswani 等人提出的一篇重要论文，介绍了变换器（Transformer）模型的架构。这篇论文彻底改变了自然语言处理（NLP）领域的研究方向，成为了许多后续模型的基础。以下是对这篇论文的详细解读：

**1. 背景**

在变换器模型出现之前，许多 NLP 任务主要依赖于循环神经网络（RNN）和长短期记忆网络（LSTM）。这些模型在处理长序列时存在一些局限性，例如训练速度慢和长距离依赖问题。

**2. 变换器模型架构**

变换器模型的核心思想是使用自注意力机制（Self-Attention）来处理输入序列。其主要组成部分包括：

**2.1 自注意力机制**

1.  自注意力：允许模型在处理单词时关注输入序列中其他单词的信息，从而捕捉到更复杂的上下文关系。
2.  计算过程：
    *   输入序列被映射为查询（Query）、键（Key）和值（Value）三种向量。
    *   通过计算查询与所有键的点积来获得注意力权重，然后将这些权重应用于值向量，得到加权和作为输出。

**2.2 多头注意力**

通过并行计算多个自注意力机制，模型可以从不同的子空间中学习信息，增强了模型的表达能力。

**2.3 前馈神经网络**

每个注意力层后面都有一个前馈神经网络，进一步处理注意力输出。

**2.4 残差连接和层归一化**

1.  每个子层（注意力层和前馈层）都有残差连接，帮助模型更好地训练。
2.  层归一化用于加速训练和提高模型稳定性。

 **3. 编码器和解码器**

1.   编码器：将输入序列转换为上下文表示，包含多个相同的层。
2.  解码器：根据编码器的输出生成目标序列，同样包含多个相同的层。

 **4. 训练过程**

1.  变换器模型使用序列到序列的学习方式，通常采用交叉熵损失函数进行训练。
2.  训练过程中使用了位置编码（Positional Encoding）来保留序列中单词的位置信息，因为自注意力机制本身没有顺序信息。 

**5. 应用**

变换器模型在机器翻译、文本生成、问答系统等多个 NLP 任务中表现出色，成为了许多后续模型（如 BERT、GPT 等）的基础。

**6. 影响**

《Attention Is All You Need》提出的变换器架构极大地推动了 NLP 领域的发展，成为了深度学习研究中的一个里程碑。它的成功也促使了其他领域（如计算机视觉）的研究者探索类似的架构。

**结论**

变换器模型通过引入自注意力机制，解决了传统 RNN 和 LSTM 模型的一些局限性，开创了新的研究方向。它的成功不仅在于模型的设计，还在于对大规模数据和计算资源的有效利用。