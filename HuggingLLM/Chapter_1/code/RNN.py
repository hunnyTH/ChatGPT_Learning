import torch.nn as nn
import torch

rnn = nn.RNN(32, 64)
input = torch.randn(4, 32)
h0 = torch.randn(1, 64)
output, hn  = rnn(input, h0)
print(output.shape, hn.shape)

wo = torch.randn(64, 1000) # 假设词表大小N=1000
logits = output @ wo  # 4×1000
probs = nn.Softmax(dim=1)(logits) # 4×1000，每一行概率和为1
print(probs)