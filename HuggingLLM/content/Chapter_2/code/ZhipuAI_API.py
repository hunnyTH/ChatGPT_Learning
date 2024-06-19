from zhipuai import ZhipuAI
 
text = "我喜欢你"

zhipu_client = ZhipuAI(api_key="e88835d488c5c1e8f922bbce6924d2ad.OB4kY1FEgGw21z4f") 
emb_req = zhipu_client.embeddings.create(
    model="embedding-2", #填写需要调用的模型名称
    input=text,
)

emb = emb_req.data[0].embedding
print(len(emb), type(emb))