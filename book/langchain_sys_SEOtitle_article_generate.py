# 用LangChain使mistral:7b-instruct-q2_K藉由對話生成文章和SEO標題，把語言模型變成愛情作家之教學

from langchain_community.llms import Ollama
from langchain_core.prompts import ChatPromptTemplate
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# 如果加上了CallbackManager就可以即時看到llm生成的文字，
# 若沒加CallbackManager則是要等到llm把文字全部生成完成後 才會顯示出來

llm = Ollama(model='mistral:7b-instruct-q2_K', callback_manager=CallbackManager([StreamingStdOutCallbackHandler()]))

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a love story creator with extensive SEO knowledge. Your task is to write an article with about 100 words, and create a SEO title for the article you wrote."),
    ("user", "{input}"),
])

# print(prompt)

chain = prompt | llm
chain.invoke({"input": input('>>> ')})

#輸入文字後，會依據文字生成SEO標題 和 文章內容