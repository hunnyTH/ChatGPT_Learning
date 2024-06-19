content = "请告诉我下面三句话的相似程度：\n1. 我喜欢你。\n2. 我钟意你。\n3.我不喜欢你。\n"
content = """请告诉我下面三句话的相似程度：
            1. 我喜欢你。
            2. 我钟意你。
            3. 我不喜欢你。
            第一句话用a表示，第二句话用b表示，第三句话用c表示。
            请以json格式输出两两语义相似度。仅输出json，不要输出其他任何内容。
            """

"""
from openai import OpenAI
OPENAI_API_KEY = "sk-fDqouTlU62yjkBhF46284543Dc8f42438a9529Df74B4Ce65"
client = OpenAI(api_key=OPENAI_API_KEY)
response = client.chat.completions.create(
    model="gpt-3.5-turbo", 
    messages=[{"role": "user", "content": content}]
)
print(response.choices[0].message.content)
"""

from zhipuai import ZhipuAI
ZHIPUAI_API_KEY = "e88835d488c5c1e8f922bbce6924d2ad.OB4kY1FEgGw21z4f"
zhipu_client = ZhipuAI(api_key=ZHIPUAI_API_KEY) 
response = zhipu_client.chat.completions.create(
    model="glm-3-turbo",
    messages=[
        {"role": "user", "content": content},
    ],
)
print(response.choices[0].message.content)