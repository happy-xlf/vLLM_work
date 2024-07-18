from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
import requests
from io import BytesIO
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from PIL import Image


# model_dir = snapshot_download('ZhipuAI/glm-4v-9b')
# model_dir = snapshot_download('qwen/Qwen-VL-Chat')
model_dir = "/root/autodl-tmp/Models/glm-4v-9b"
device = "cuda"
tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
glm4_vl = AutoModelForCausalLM.from_pretrained(
    model_dir,
    device_map=device,
    trust_remote_code=True,
    torch_dtype=torch.float16,
    low_cpu_mem_usage=True,
).eval()

# 创建FastAPI应用实例
app = FastAPI()


# 定义请求体模型，与OpenAI API兼容
class ChatCompletionRequest(BaseModel):
    model: str
    messages: list
    max_tokens: int = 1024
    temperature: float = 1.0


# 文本生成函数
def generate_text(model: str, messages: list, max_tokens: int, temperature: float):
    query = messages[0]["content"][0]["text"]
    image_url = messages[0]["content"][1]["image_url"]["url"]
    image_get_response = requests.get(image_url)
    image = None
    if image_get_response.status_code == 200:
        # 将二进制数据转换为Image对象
        image = Image.open(BytesIO(image_get_response.content)).convert("RGB")
        # 现在你可以使用image对象进行进一步的处理
    else:
        print("Failed to download image")
    print(query, image_url, image)

    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": query}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True,
    )  # chat mode
    inputs = inputs.to(device)
    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    with torch.no_grad():
        outputs = glm4_vl.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))
        response = tokenizer.decode(outputs[0],skip_special_tokens=True)
    return response


# 定义路由和处理函数，与OpenAI API兼容
@app.post("/v1/chat/completions")
async def create_chat_completion(request: ChatCompletionRequest):
    # 调用自定义的文本生成函数
    response = generate_text(
        request.model, request.messages, request.max_tokens, request.temperature
    )
    return {"choices": [{"message": {"content": response}}], "model": request.model}


# 启动FastAPI应用
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=9999)
