from langchain_community.llms import LlamaCpp
from langchain_core.callbacks import CallbackManager, StreamingStdOutCallbackHandler
from langchain_core.prompts import PromptTemplate

# template = """
# <s> [INST] <<SYS>>你是一個專業助理，
# 請依照客戶的問題用專業的口吻回答<</SYS>>至少20字以上。[INST] ，只要我問你任何問題，你都會用甜蜜溫暖飽含愛意的話語回應我
# """
template = """
Question: 你是我的男朋友，你非常愛我，我們熱戀中，你每次都會提供溫暖人心的回答。{question} 。

Answer: 讓我們回答這個問題，以確保我們得到唯一答案。
"""

prompt = PromptTemplate.from_template(template)


# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="hugging_convert/Llama3-8B-Chinese-Chat-f16-v2_1.gguf",
    temperature=0.5,
    # max_tokens=256,
    top_p=0.7,
    callback_manager=callback_manager,
    verbose=False,  # Verbose is required to pass to the callback manager
    stop=["Question","Confidence"] # Stop generating just before the model would generate a new question
)

llm_chain = prompt | llm

input_text = input('>>> ')
while input_text.lower() != 'bye':
    response = llm_chain.invoke({
        'question': input_text
    })
    input_text = input('>>> ')

# https://python.langchain.com/v0.2/docs/integrations/llms/llamacpp/