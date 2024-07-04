from dataclasses import dataclass
import pandas as pd
df = pd.read_csv(r"HuggingLLM\content\Chapter_2\data\train.csv")
print(df.shape)
print(df.head())
print(df["Class Index"].value_counts())

# 指定种子，不然得不到下面一样的结果
sdf = df.sample(100, random_state=26187)
print(sdf["Class Index"].value_counts())
print(sdf.iloc[1])

# 维护一个用户偏好和行为记录
from typing import List
@dataclass
class User:
    
    user_name: str

@dataclass
class UserPrefer:
    
    user_name: str
    prefers: List[int]


@dataclass
class Item:
    
    item_id: str
    item_props: dict


@dataclass
class Action:
    
    action_type: str
    action_props: dict


@dataclass
class UserAction:
    
    user: User
    item: Item
    action: Action
    action_time: str

u1 = User("u1")
up1 = UserPrefer("u1", [1, 2])
print(sdf.iloc[1])
i1 = Item("i1", {
    "id": 1, 
    "catetory": "sport",
    "title": "Swimming: Shibata Joins Japanese Gold Rush", 
    "description": "\
    ATHENS (Reuters) - Ai Shibata wore down French teen-ager  Laure Manaudou to win the women's 800 meters \
    freestyle gold  medal at the Athens Olympics Friday and provide Japan with  their first female swimming \
    champion in 12 years.", 
    "content": "content"
})
a1 = Action("浏览", {
    "open_time": "2023-04-01 12:00:00", 
    "leave_time": "2023-04-01 14:00:00",
    "type": "close",
    "duration": "2hour"
})
ua1 = UserAction(u1, i1, a1, "2023-04-01 12:00:00")

# 计算文本Embedding
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from zhipuai import ZhipuAI
ZHIPUAI_API_KEY = "e88835d488c5c1e8f922bbce6924d2ad.OB4kY1FEgGw21z4f"
zhipu_client = ZhipuAI(api_key=ZHIPUAI_API_KEY) 
def get_embedding_zhipu(text):
    emb_req = zhipu_client.embeddings.create(
        model="embedding-2",
        input=text,
    )
    return emb_req.data[0].embedding

sdf["embedding"] = sdf.apply(lambda x: get_embedding_zhipu(x.Title + x.Description), axis=1)

import random
class Recall:
    
    def __init__(self, df: pd.DataFrame):
        self.data = df
    
    def user_prefer_recall(self, user, n):
        up = self.get_user_prefers(user)
        idx = random.randrange(0, len(up.prefers))
        return self.pick_by_idx(idx, n)
    
    def hot_recall(self, n):
        # 随机进行示例
        df = self.data.sample(n)
        return df
    
    def user_action_recall(self, user, n):
        actions = self.get_user_actions(user)
        interest = self.get_most_interested_item(actions)
        recoms = self.recommend_by_interest(interest, n)
        return recoms
    
    def get_most_interested_item(self, user_action):
        """
        可以选近一段时间内用户交互时间、次数、评论（相关属性）过的Item
        """
        # 就是sdf的第2行，idx为1的那条作为最喜欢（假设）
        # 是一条游泳相关的Item
        idx = user_action.item.item_props["id"]
        im = self.data.iloc[idx]
        return im
    
    def recommend_by_interest(self, interest, n):
        cate_id = interest["Class Index"]
        q_emb = interest["embedding"]
        # 确定类别
        base = self.data[self.data["Class Index"] == cate_id]
        # 此处可以复用QA那一段代码，用给定embedding计算base中embedding的相似度
        base_arr = np.array(
            [v.embedding for v in base.itertuples()]
        )
        q_arr = np.expand_dims(q_emb, 0)
        sims = cosine_similarity(base_arr, q_arr)
        # 排除掉自己
        idxes = sims.argsort(0).squeeze()[-(n+1):-1]
        return base.iloc[reversed(idxes.tolist())]
    
    def pick_by_idx(self, category, n):
        df = self.data[self.data["Class Index"] == category]
        return df.sample(n)
    
    def get_user_actions(self, user):
        dct = {"u1": ua1}
        return dct[user.user_name]
    
    def get_user_prefers(self, user):
        dct = {"u1": up1}
        return dct[user.user_name]
    
    def run(self, user):
        ur = self.user_action_recall(user, 5)
        if len(ur) == 0:
            ur = self.user_prefer_recall(user, 5)
        hr = self.hot_recall(3)
        return pd.concat([ur, hr], axis=0)       

r = Recall(sdf)
rd = r.run(u1)
print(rd)