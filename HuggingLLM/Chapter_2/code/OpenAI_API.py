from openai import OpenAI
"""
API 网址
https://free.gpt.ge
API 密钥
sk-bRrj7q3Obtl8iISi72816711D0404323942e885bD6352d87
"""

OPENAI_API_KEY = "sk-bRrj7q3Obtl8iISi72816711D0404323942e885bD6352d87"
client = OpenAI(api_key=OPENAI_API_KEY)

text = "我喜欢你"
model = "text-embedding-ada-002"

emb_req = client.embeddings.create(input=[text], model=model)

emb = emb_req.data[0].embedding
print(len(emb), type(emb))