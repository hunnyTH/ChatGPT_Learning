import pandas as pd
import numpy as np
df = pd.read_csv("HuggingLLM\content\Chapter_2\data\DBPEDIA_val.csv")
"""
print(df.shape)
print(df.head())
print(df.l1.value_counts())
"""

# 随机取200条数据
sdf = df.sample(200)
print(sdf.l1.value_counts())

cdf = sdf[
    (sdf.l1 == "Place") | (sdf.l1 == "Work") | (sdf.l1 == "Species")
]
print(cdf.shape)

from zhipuai import ZhipuAI
ZHIPUAI_API_KEY = "e88835d488c5c1e8f922bbce6924d2ad.OB4kY1FEgGw21z4f"
zhipu_client = ZhipuAI(api_key=ZHIPUAI_API_KEY) 
def get_embedding_zhipu(text):
    emb_req = zhipu_client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return emb_req.data[0].embedding

# 将文本变为向量
cdf["embedding"] = cdf.text.apply(lambda x: get_embedding_zhipu(x))

# 使用PCA（主成分分析）进行降维
from sklearn.decomposition import PCA
arr = np.array(cdf.embedding.tolist())
pca = PCA(n_components=3)
vis_dims = pca.fit_transform(arr)
cdf["embed_vis"] = vis_dims.tolist()
print(arr.shape)
print(vis_dims.shape)

import matplotlib.pyplot as plt

fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
cmap = plt.get_cmap("tab20")
categories = sorted(cdf.l1.unique())

# 分别绘制每个类别
for i, cat in enumerate(categories):
    sub_matrix = np.array(cdf[cdf.l1 == cat]["embed_vis"].to_list())
    x=sub_matrix[:, 0]
    y=sub_matrix[:, 1]
    z=sub_matrix[:, 2]
    colors = [cmap(i/len(categories))] * len(sub_matrix)
    ax.scatter(x, y, z, c=colors, label=cat)

ax.legend(bbox_to_anchor=(1.2, 1))
plt.show();