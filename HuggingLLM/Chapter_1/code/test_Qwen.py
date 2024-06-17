# qwen
from http import HTTPStatus
import dashscope

DASHSCOPE_API_KEY="" # 请填写您自己的APIKey

messages = [{'role': 'system', 'content': 'You are a helpful assistant.'},
            {'role': 'user', 'content': '请你介绍一下Datawhale。'}]

responses = dashscope.Generation.call(
    dashscope.Generation.Models.qwen_max, # 请选择参考官方文档，填写需要调用的模型名称
    api_key=DASHSCOPE_API_KEY, 
    messages=messages,
    result_format='message',  # 将结果设置为“消息”格式
    stream=True, #流式输出
    incremental_output=True  
)


full_content = ''  # 合并输出
for response in responses:
    if response.status_code == HTTPStatus.OK:
        full_content += response.output.choices[0]['message']['content']
        # print(response)
    else:
        print('Request id: %s, Status code: %s, error code: %s, error message: %s' % (
            response.request_id, response.status_code,
            response.code, response.message
        ))
print('回答:\n' + full_content)
