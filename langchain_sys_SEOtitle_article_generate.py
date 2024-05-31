from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 如果加上了CallbackManager就可以即時看到llm生成的文字，
# 若沒加CallbackManager則是要等到llm把文字全部生成完成後 才會顯示出來

llm = Ollama(model='llama3', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a love story creator with extensive SEO knowledge. Your task is to write an article with about 100 words, and create a SEO title for the article you wrote."),
    ("user", "{input}"),
])

chain = prompt | llm
print(chain.invoke({"input": input('>>> ')}))

# from langchain.callbacks.manager import CallbackManager
# from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
# from langchain_community.llms import Ollama

# llm = Ollama(
#     model="llama2", callback_manager=CallbackManager([StreamingStdOutCallbackHandler()])
# )
# input_text = input('>>> ')
# print("wellcome to lotus world")
# while input_text.lower() != 'bye':
#     input_text=input('>>> ')
#     llm.invoke(input_text)

# ---------------
# from langchain_community.llms import Ollama
# from langchain_core.prompts import ChatPromptTemplate

# llm = Ollama(model='llama2' )

# prompt = ChatPromptTemplate.from_messages([
#     ("user", "{input}"),
# ])

# chain = prompt | llm

# print(chain.invoke({"input": "Hi, how are you today?"}))

# ---------------


# root@4be643ba6a94:/app# python3 langchain_sys_SEOtitle_article_generate.py
# >>> love
# **Title:** The Power of Unconditional Love: How It Can Transform Your Life

# **Article:**

# Unconditional love has the power to transform our lives in profound ways. When we love someone without condition, we open ourselves up to a deeper connection and understanding. This type of love is not based on what someone does for us, but rather who they are as a person. It's a choice to accept and cherish them just as they are. By practicing unconditional love, we can build stronger relationships, cultivate empathy and compassion, and even improve our mental health. In a world that often values conditionality, it's essential to remember the transformative power of loving someone without expectation or attachment.

# **SEO Title:** "The Transformative Power of Unconditional Love: Boosting Mental Health and Building Stronger Relationships"

# Keywords: unconditional love, transformative power, mental health, relationships, self-acceptance.


# https://myapollo.com.tw/blog/langchain-tutorial-get-started/
