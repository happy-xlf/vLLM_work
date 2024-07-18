#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :client.py
# @Time      :2024/07/15 17:43:18
# @Author    :Lifeng
# @Description :
from openai import OpenAI
client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="token-abc123",
)

completion = client.chat.completions.create(
    model="Qwen1.5-4B-Chat",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "计算999+101=？"}
    ],
    max_tokens=1024,
    temperature=0.9,
    top_p=0.7,
)

print(completion.choices[0].message)