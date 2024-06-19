import pandas as pd
import numpy as np
df = pd.read_csv("HuggingLLM\content\Chapter_2\data\Kaggle related questions on Qoura - Questions.csv")
print(df.shape)
print(df.head())

"""
把Link当做答案构造数据对。基本的流程如下：
- 对每个Question计算Embedding
- 存储Embedding，同时存储每个Question对应的答案
- 从存储的地方检索最相似的Question
"""
# 计算两个向量的余弦相似度
def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

query = "is kaggle alive?"
"""
vec_base = []
from openai import OpenAI
OPENAI_API_KEY = "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"
open_client = OpenAI(api_key=OPENAI_API_KEY)
def get_embedding_open(text,model = "text-embedding-ada-002"):
    emb_req = open_client.embeddings.create(model=model,input = text)
    return emb_req.data[0].embedding
for v in df.head().itertuples():
    emb = get_embedding_open(v.Questions)
    im = {
        "question": v.Questions,
        "embedding": emb,
        "answer": v.Link
    }
    vec_base.append(im)
q_emb = get_embedding_open(query)
sims = [cosine_similarity(q_emb, v["embedding"]) for v in vec_base]
print(sims)
"""

from zhipuai import ZhipuAI
ZHIPUAI_API_KEY = "e88835d488c5c1e8f922bbce6924d2ad.OB4kY1FEgGw21z4f"
zhipu_client = ZhipuAI(api_key=ZHIPUAI_API_KEY) 
def get_embedding_zhipu(text):
    emb_req = zhipu_client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return emb_req.data[0].embedding
vec_base_zhipu = []
for v in df.head().itertuples():
    emb = get_embedding_zhipu(v.Questions)
    im = {
        "question": v.Questions,
        "embedding": emb,
        "answer": v.Link
    }
    vec_base_zhipu.append(im)

q_emb_zhpu = get_embedding_zhipu(query)
sims_zhipu = [cosine_similarity(q_emb_zhpu, v["embedding"]) for v in vec_base_zhipu]
print(sims_zhipu)

print(
    #vec_base[1]["question"], vec_base[1]["answer"], 
    vec_base_zhipu[1]["question"], vec_base_zhipu[1]["answer"]
)

# 使用NumPy进行批量计算
arr = np.array(
    [v["embedding"] for v in vec_base_zhipu]
)
print(arr.shape)
q_arr = np.expand_dims(q_emb_zhpu, 0)
print(q_arr.shape)

from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(arr, q_arr))

