import torch
from transformers import pipeline, StoppingCriteria

# Define a custom stopping criteria class
class EosListStoppingCriteria(StoppingCriteria):
    def __init__(self, eos_sequence=[128256]):
        self.eos_sequence = eos_sequence

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        last_ids = input_ids[:, -len(self.eos_sequence):].tolist()
        return self.eos_sequence in last_ids

# Initialize the model with automatic device mapping
llm = pipeline("text-generation", model="Llama-3-Taiwan-8B-Instruct", device_map="auto")
tokenizer = llm.tokenizer

# Define a conversation example
chat = [
    {"role": "system", "content": "You are an AI assistant called Twllm, created by TAME (TAiwan Mixture of Expert) project."},
    {"role": "user", "content": "你好，請問你可以完成什麼任務？"},
    {"role": "assistant", "content": "你好，我可以幫助您解決各種問題、提供資訊並協助完成多種任務。例如：回答技術問題、提供建議、翻譯文字、尋找資料或協助您安排行程等。請告訴我如何能幫助您。"},
    {"role": "user", "content": "告訴我台灣的經濟如何？"}
]
flatten_chat_for_generation = tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)


# Generate a response using the custom stopping criteria
output = llm(flatten_chat_for_generation, return_full_text=False, max_new_tokens=128, top_p=0.9, temperature=0.7, stopping_criteria=[EosListStoppingCriteria([tokenizer.eos_token_id])])
print(output[0]['generated_text'])
"謝謝！很高興能夠為您服務。如果有任何其他需要協助的地方，請隨時與我聯繫。我會盡最大努力為您提供所需的支援。"
