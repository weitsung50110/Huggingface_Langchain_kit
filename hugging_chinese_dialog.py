# from dialogbot import GPTBot
# model = GPTBot("~/trans_project/hugging_convert/gpt2-dialogbot-base-chinese")
# r = model.answer("今天你的病好点了吗？")
from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "hugging_convert/Llama3-8B-Chinese-Chat"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

messages = [
    {"role": "user", "content": "7年前，妈妈年龄是儿子的6倍，儿子今年12岁，妈妈今年多少岁？"},
]

input_ids = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, return_tensors="pt"
).to(model.device)

outputs = model.generate(
    input_ids,
    max_new_tokens=8192,
    do_sample=True,
    temperature=0.6,
    top_p=0.9,
)
response = outputs[0][input_ids.shape[-1]:]
print(tokenizer.decode(response, skip_special_tokens=True))

# https://huggingface.co/shenzhi-wang/Llama3-8B-Chinese-Chat