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

# 使用循环进行计算
for v in df.head().itertuples():
    emb = get_embedding_zhipu(v.Questions)
    im = {
        "question": v.Questions,
        "embedding": emb,
        "answer": v.Link
    }
    vec_base_zhipu.append(im)

q_emb_zhipu = get_embedding_zhipu(query)
sims_zhipu = [cosine_similarity(q_emb_zhipu, v["embedding"]) for v in vec_base_zhipu]
print(sims_zhipu)
print(vec_base_zhipu[1]["question"], vec_base_zhipu[1]["answer"])

# 使用NumPy进行批量计算
arr = np.array([v["embedding"] for v in vec_base_zhipu])
print(arr.shape)
q_arr = np.expand_dims(q_emb_zhipu, 0)
print(q_arr.shape)

from sklearn.metrics.pairwise import cosine_similarity
print(cosine_similarity(arr, q_arr))

# 使用语义检索工具
import redis
r = redis.Redis()   # 连接到服务器，并创建实例
r.set("key", "value")
print(r.get("key"))

VECTOR_DIM = 1024
INDEX_NAME = "faq"  # 语义检索索引

from redis.commands.search.query import Query
from redis.commands.search.field import TextField, VectorField
from redis.commands.search.indexDefinition import IndexDefinition

# 建好要存字段的索引，针对不同属性字段，使用不同Field
question = TextField(name="question")   # 问题的文本和向量表示
answer = TextField(name="answer")       # 答案的文本和向量表示
embedding = VectorField(
    name="embedding", 
    algorithm="HNSW",   # Hierarchical Navigable Small Worlds，HNSW，一种高效的相似搜索算法。
    attributes={
        "TYPE": "FLOAT32",
        "DIM": VECTOR_DIM,
        "DISTANCE_METRIC": "COSINE"
    }
)
schema = (question, embedding, answer)
index = r.ft(INDEX_NAME)
try:
    info = index.info()
except:
    index.create_index(schema, definition=IndexDefinition(prefix=[INDEX_NAME + "-"]))

# 如果需要删除已有文档的话，可以使用下面的命令
# index.dropindex(delete_documents=True)

# 将数据存入Redis
for v in df.head().itertuples():
    # 智谱AI
    emb = get_embedding_zhipu(v.Questions)
    # 注意，redis要存储bytes或string
    emb = np.array(emb, dtype=np.float32).tobytes() # 转为字节数组
    im = {
        "question": v.Questions,
        "embedding": emb,
        "answer": v.Link
    }
    # 使用hest命令将数据存储在索引中
    r.hset(name=f"{INDEX_NAME}-{v.Index}", mapping=im)  

# 构造查询输入
query = "kaggle alive?"
# 智谱AI
embed_query = get_embedding_zhipu(query)
# 将使用 np.array 将查询的向量表示转换为字节数组，并将其存储在一个字典中。
params_dict = {"query_embedding": np.array(embed_query).astype(dtype=np.float32).tobytes()}

# 定义基本查询，包括要使用的距离函数、要检索的向量字段、要查询的向量表示和要返回的字段。
# 在本例中，我们将使用余弦相似度作为距离函数，
# 使用 question 和 embedding 字段来检索向量表示，
# 使用 score 字段来计算相似度，并返回 question、 answer 和 score 字段。
k = 3
# {some filter query}=>[ KNN {num|$num} @vector_field $query_vec]
base_query = f"* => [KNN {k} @embedding $query_embedding AS score]"
return_fields = ["question", "answer", "score"]
query = (
    Query(base_query)
     .return_fields(*return_fields)
     .sort_by("score")
     .paging(0, k)
     .dialect(2)
)
# 使用 search 方法执行查询，并获取结果。
# 在本例中，我们将返回前 k 个结果，并打印结果的 ID、问题和答案以及相似度。
res = index.search(query, params_dict)
for i,doc in enumerate(res.docs):
    similarity = 1 - float(doc.score)
    print(f"{doc.id}, {doc.question}, {doc.answer} (Similarity: {round(similarity ,3) })")