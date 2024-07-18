#!/usr/bin/env python
# -*- coding:utf-8 -*-
# @FileName  :vllm_qwen_infer.py
# @Time      :2024/07/15 17:18:05
# @Author    :Lifeng
# @Description :
from vllm import LLM
from vllm.sampling_params import SamplingParams


class QwenVllm(object):
    def __init__(self, gpu_num=2, max_tokens=512):
        self.gpu_num = gpu_num
        self.max_tokens = max_tokens
        # self.model_path= "../../Models/Qwen1.5-4B-Chat/"
        self.model_path= "/root/autodl-tmp/Models/Qwen1.5-4B-Chat"
        self.model, self.tokenizer = self.model_load_with_vllm()
 
 
    def model_load_with_vllm(self):
        """
        vllm 形式预加载 模型 
        """ 
        model = LLM(
            tokenizer=self.model_path,
            model=self.model_path,
            dtype="bfloat16",
            tokenizer_mode= 'slow',
            trust_remote_code=True,
            tensor_parallel_size=self.gpu_num,
            gpu_memory_utilization=0.8, # gpu 初始化显存占比，这里单卡48g显存
            max_seq_len_to_capture=8192,
            max_model_len = 8192
        )

        tokenizer = model.get_tokenizer()

        return model, tokenizer
 
 
    def qwen_chat_vllm(self, prompt):
        """ vllm batch推理注意 batch size 与 gpu 关系"""
 
        message= [
            {"role": "system", "content": "you are a great assistant."},
            {"role": "user", "content": prompt}
        ]
 
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
 
        # max_token to control the maximum output length
        sampling_params = SamplingParams(
            temperature=0.7,
            top_p=0.8,
            repetition_penalty=1.2,
            max_tokens=self.max_tokens)
 
        outputs = self.model.generate([text], sampling_params)
 
        response = []
        for output in outputs:
            # prompt = output.prompt
            generated_text = output.outputs[0].text
            response.append(generated_text)
 
        return response
 
if __name__ == '__main__':
    run = QwenVllm(gpu_num=1, max_tokens=1024) 
    # 大模型单论对话生成
    prompt = """# 写一个快速排序代码"""
    response = run.qwen_chat_vllm(prompt=prompt)
    print(response)
