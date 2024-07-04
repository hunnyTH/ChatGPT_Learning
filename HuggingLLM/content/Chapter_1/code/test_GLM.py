# GLM
from zhipuai import ZhipuAI

client = ZhipuAI(api_key="e88835d488c5c1e8f922bbce6924d2ad.OB4kY1FEgGw21z4f") # 请填写您自己的APIKey

messages = [{"role": "system", "content": "你是一个乐于解答各种问题的助手，你的任务是为用户提供专业、准确、有见地的建议。"},
            {"role": "user", "content": "请你介绍一下Datawhale。"},]

response = client.chat.completions.create(
    model="glm-4",  # 请选择参考官方文档，填写需要调用的模型名称
    messages=messages, # 将结果设置为“消息”格式
    stream=True,  # 流式输出
)

full_content = ''  # 合并输出
for chunk in response:
    full_content += chunk.choices[0].delta.content
print('回答:\n' + full_content)