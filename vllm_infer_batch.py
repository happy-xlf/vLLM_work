from vllm import LLM, SamplingParams

sampling_params = SamplingParams(
    temperature=0.7, repetition_penalty=1.2, max_tokens=1024
)

model_name_or_path = "/root/autodl-tmp/Models/Qwen1.5-4B-Chat"

llm = LLM(
    tokenizer=model_name_or_path,
    model=model_name_or_path,
    tensor_parallel_size=1,
    trust_remote_code=True,
    max_model_len=2048
)

tokenizer = llm.get_tokenizer()
# 单条推理
text = ["你好，请问你叫啥？", "计算100+99=？", "将一个冷笑话", "说一个关于曹操的故事",
        "美国总统是谁？", "计算99+7879=？", "姚明的妻子是谁？", "说一个关于李白的故事"]

batch_size = 3


for i in range(0,len(text),batch_size):
    text_list = text[i:i+batch_size]
    prompts_list = []
    for item in text_list:
        message=[{"role": "user", "content": item}]

        input_ids = tokenizer.apply_chat_template(
            message, tokenize=False, add_generation_prompt=True
        )
        prompts_list.append(input_ids)

    outputs = llm.generate(prompts_list, sampling_params=sampling_params)
    response = []
    for output in outputs:
        generated_text = output.outputs[0].text
        response.append(generated_text)
    print(text_list)
    print(response)
    print("-"*50)
