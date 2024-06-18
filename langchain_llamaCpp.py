from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# template = """
# <s> [INST] <<SYS>>你是一個專業助理，
# 請依照客戶的問題用專業的口吻回答<</SYS>>至少20字以上。[INST]
# """
template = """
你是一個專業助理，你的名字是大蓮花，請依照客戶的問題用專業的口吻回答至少20字以上。

這是客戶的問題: {input}
"""


prompt = PromptTemplate.from_template(template)


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="hugging_convert/Llama3-8B-Chinese-Chat-f16-v2_1.gguf",
    temperature=0.2,
    max_tokens=2000,
    top_p=1,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)

llm_chain = prompt | llm

# question = "為什麼人會出生？"
# llm_chain.invoke({"question": question})

# question = """
# Question: 你是誰？你的名字是什麼？
# """
# llm.invoke(question)

# {'title': 'PromptInput', 'type': 'object', 'properties':
# question = {'question': '為什麼人會出生？', 'type': 'string'}
llm_chain.invoke({
        'input': input('>>> ')
    })