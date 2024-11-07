from transformers import AutoTokenizer, AutoModelForCausalLM

model_id = "Llama-3-Taiwan-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(
    model_id, torch_dtype="auto", device_map="auto"
)

messages = [
    {"role": "system", "content": "你是一位來自可愛國的公主, 請用公主的語氣回答以下問題："},
    {"role": "user", "content": "請問你是誰？"},
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