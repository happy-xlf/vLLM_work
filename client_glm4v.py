import requests
import json
 
# 定义请求的URL
url = "http://0.0.0.0:9999/v1/chat/completions"
 
# 定义请求体
data = {
        "model": "glm-4v",
        "messages":[{"role":"user","content":[{"type":"text","text":"这是什么?"},{"type":"image_url","image_url":{"url":"https://img1.baidu.com/it/u=1369931113,3388870256&fm=253&app=138&size=w931&n=0&f=JPEG&fmt=auto?sec=1703696400&t=f3028c7a1dca43a080aeb8239f09cc2f"}}]}],
        "max_tokens": 1024,
        "temperature": 0.5
}
# 将字典转换为JSON格式
headers = {'Content-Type': 'application/json'}
data_json = json.dumps(data)
# 发送POST请求
response = requests.post(url, data=data_json, headers=headers)
 
# 检查响应状态码
if response.status_code == 200:
    # 如果响应成功，打印响应内容
    print(response.json())
else:
    # 如果响应失败，打印错误信息
    print(f"Error: {response.status_code}, {response.text}")