from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# template = """
# <s> [INST] <<SYS>>你是一個專業助理，
# 請依照客戶的問題用專業的口吻回答<</SYS>>至少20字以上。[INST]
# """
template = """Question: {question}

Answer: 讓我們解決這個問題，以確保我們得到正確的答案"""

prompt = PromptTemplate.from_template(template)


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="hugging_convert/Llama3-8B-Chinese-Chat-f16-v2_1.gguf",
    temperature=0,
    # max_tokens=256,
    top_p=0.9,
    callback_manager=callback_manager,
    verbose=False,  # Verbose is required to pass to the callback manager
)

llm_chain = prompt | llm

# question = "為什麼人會出生？"
# llm_chain.invoke({"question": question})

input_text = input('>>> ')
while input_text.lower() != 'bye':
    response = llm_chain.invoke({
        'question': input_text
    })
    input_text = input('>>> ')

# {'title': 'PromptInput', 'type': 'object', 'properties':
# question = {'question': '為什麼人會出生？', 'type': 'string'}
# input_text = input('>>> ')
# while input_text.lower() != 'bye':
#     response = llm_chain.invoke({
#         'input': input_text
#     })


'''Temperature
Temperature 調節語言模型輸出的不可預測性。透過較高的溫度設置，輸出變得更具創造性且更難以預測，
因為它放大了不太可能的令牌的可能性，同時降低了更可能的令牌的可能性。
相反，較低的溫度會產生更保守和可預測的結果。以下範例說明了輸出中的這些差異。

Top P[^b] 是語言模型中的一項可設定的超參數，有助於管理其輸出的隨機性。它的工作原理是建立一個機率閾值，
然後選擇組合可能性超過此限制的令牌。

https://learnprompting.org/zh-tw/docs/basics/configuration_hyperparameters

top-k 与 top-p 为选择 token 引入了随机性，让其他高分的 token 有被选择的机会，不像 greedy decoding 一样总是选最高分的。
https://juejin.cn/post/7236558485290631205
'''

# https://python.langchain.com/v0.2/docs/integrations/llms/llamacpp/