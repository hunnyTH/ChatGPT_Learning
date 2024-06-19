# 余弦相似度
import numpy as np
a = [0.1, 0.2, 0.3]
b = [0.2, 0.3, 0.4]
cosine_ab = (0.1*0.2+0.2*0.3+0.3*0.4)/(np.sqrt(0.1**2+0.2**2+0.3**2) * np.sqrt(0.2**2+0.3**2+0.4**2))
print(cosine_ab)

# 计算两个向量的余弦相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

from openai import OpenAI
OPENAI_API_KEY = "填入专属的API key"
open_client = OpenAI(api_key=OPENAI_API_KEY)
def get_embedding_open(text,model = "text-embedding-ada-002"):
    emb_req = open_client.embeddings.create(model=model,input = text)
    return emb_req.data[0].embedding

from zhipuai import ZhipuAI
ZHIPUAI_API_KEY = "填入专属的API key"
zhipu_client = ZhipuAI(api_key=ZHIPUAI_API_KEY) 
def get_embedding_zhipu(text):
    emb_req = zhipu_client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return emb_req.data[0].embedding

# 注意它支持多种模型，可以通过接口查看
text1 = "我喜欢你"
text2 = "我钟意你"
text3 = "我不喜欢你"
emb1 = get_embedding_open(text1, "text-embedding-3-large")
emb2 = get_embedding_open(text2, "text-embedding-3-large")
emb3 = get_embedding_open(text3, "text-embedding-3-large")

emb1_zhipu = get_embedding_zhipu(text1)
emb2_zhipu = get_embedding_zhipu(text2)
emb3_zhipu = get_embedding_zhipu(text3)

print(len(emb1), type(emb1), len(emb1_zhipu), type(emb1_zhipu))
print(cosine_similarity(emb1, emb2), cosine_similarity(emb1_zhipu, emb2_zhipu))
print(cosine_similarity(emb1, emb3), cosine_similarity(emb1_zhipu, emb3_zhipu))
print(cosine_similarity(emb2, emb3), cosine_similarity(emb2_zhipu, emb3_zhipu))

# OpenAI 其他Embedding模型
text1 = "我喜欢你"
text2 = "我钟意你"
text3 = "我不喜欢你"
emb1 = get_embedding_open(text1, "text-embedding-ada-002")
emb2 = get_embedding_open(text2, "text-embedding-ada-002")
emb3 = get_embedding_open(text3, "text-embedding-ada-002")
print(cosine_similarity(emb1, emb2))
print(cosine_similarity(emb1, emb3))
print(cosine_similarity(emb2, emb3))
